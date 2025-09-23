import cv2
import numpy as np
from flask_socketio import emit
from datetime import datetime
from PIL import Image
import logging
import io
import os
from pathlib import Path
from config.settings import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightImageDetector:
    def __init__(self):
        logger.info("Initializing lightweight image detector...")
        # No heavy models to load!
        self.ml_classifier = None
        self.feature_scaler = None
        self._load_ml_classifier()
    
    def _load_ml_classifier(self):
        """Load optional ML classifier if available and enabled."""
        try:
            if getattr(Config, 'ML_CLASSIFIER_ENABLED', 'false').lower() == 'true':
                import joblib
                model_dir = Path(__file__).parent.parent.parent / "models"
                
                model_path = model_dir / "image_classifier.joblib"
                scaler_path = model_dir / "feature_scaler.joblib"
                
                if model_path.exists() and scaler_path.exists():
                    self.ml_classifier = joblib.load(model_path)
                    self.feature_scaler = joblib.load(scaler_path)
                    logger.info("ML classifier loaded successfully")
                else:
                    logger.info("ML classifier files not found, using rule-based detection only")
        except Exception as e:
            logger.warning(f"Failed to load ML classifier: {e}")
    
    def _extract_feature_vector(self, features, dct_stats, noise_stats, statistical_stats, metadata_stats, ela, exif):
        """Extract feature vector for ML classifier."""
        feature_names = [
            'edge_density', 'texture_variance', 'color_uniformity', 'freq_variance',
            'avg_compression', 'compression_var', 'ela_score', 'exif_present',
            'dct_anomaly_ratio', 'noise_gaussian_deviation', 'noise_spatial_consistency',
            'gray_entropy', 'gradient_skewness', 'camera_metadata_completeness',
            'bytes_per_pixel', 'suspicious_score_raw', 'green_score'
        ]
        
        # Combine all feature sources
        all_features = {**features, **dct_stats, **noise_stats, **statistical_stats, **metadata_stats}
        all_features['ela_score'] = ela
        all_features['exif_present'] = float(exif.get('has_exif', False))
        
        feature_vector = []
        for name in feature_names[:-2]:  # Exclude suspicious_score_raw and green_score for now
            value = all_features.get(name, 0)
            if isinstance(value, bool):
                value = float(value)
            feature_vector.append(value)
        
        return np.array(feature_vector)

    def _ml_predict(self, feature_vector, suspicious_score, green_score):
        """Use ML classifier to make prediction."""
        try:
            # Add the rule-based scores as features
            extended_vector = np.append(feature_vector, [suspicious_score, green_score])
            extended_vector = extended_vector.reshape(1, -1)
            
            # Scale features
            scaled_features = self.feature_scaler.transform(extended_vector)
            
            # Get prediction and probability
            prediction = self.ml_classifier.predict(scaled_features)[0]
            probability = self.ml_classifier.predict_proba(scaled_features)[0]
            
            # probability[1] is the probability of being fake
            fake_probability = probability[1]
            
            return bool(prediction), float(fake_probability)
            
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return None, None
    
    # ---------- Feature extraction helpers ----------
    def _extract_traditional_features(self, image_path):
        """Extract features using only OpenCV and numpy"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            features = {}
            
            # 1. Edge density analysis (deepfakes often have inconsistent edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
            features['edge_density'] = edge_density
            
            # 2. Texture analysis using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features['texture_variance'] = laplacian_var
            
            # 3. Color distribution analysis
            color_std = np.std(image.reshape(-1, 3), axis=0)
            features['color_uniformity'] = np.mean(color_std)
            
            # 4. Brightness distribution
            brightness_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            brightness_entropy = -np.sum(brightness_hist * np.log2(brightness_hist + 1e-10))
            features['brightness_entropy'] = brightness_entropy
            
            # 5. Local Binary Pattern-like texture
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            texture_response = cv2.filter2D(gray, -1, kernel)
            features['local_texture_var'] = np.var(texture_response)
            
            # 6. Frequency domain analysis (simplified)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            features['freq_energy'] = np.mean(magnitude_spectrum)
            features['freq_variance'] = np.var(magnitude_spectrum)

            # 7. Simple quality metrics
            features['brightness_mean'] = float(np.mean(gray))
            features['brightness_std'] = float(np.std(gray))
            features['width'] = int(image.shape[1])
            features['height'] = int(image.shape[0])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}
    
    def _analyze_compression_artifacts(self, image_path):
        """Detect compression artifacts that might indicate manipulation"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply DCT-like analysis to detect compression patterns
            # Split image into 8x8 blocks (like JPEG)
            h, w = gray.shape
            block_size = 8
            compression_scores = []
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size].astype(np.float32)
                    
                    # Simple frequency analysis of block
                    fft_block = np.fft.fft2(block)
                    high_freq = np.sum(np.abs(fft_block[4:, 4:]))  # High frequency components
                    total_energy = np.sum(np.abs(fft_block))
                    
                    if total_energy > 0:
                        hf_ratio = high_freq / total_energy
                        compression_scores.append(hf_ratio)
            
            avg_compression = np.mean(compression_scores)
            compression_var = np.var(compression_scores)
            
            return avg_compression, compression_var
            
        except Exception as e:
            logger.error(f"Error analyzing compression: {str(e)}")
            return 0.1, 0.01
    
    def _detect_face_inconsistencies(self, image_path):
        """Simple face detection and consistency check. Returns (issues, face_count)."""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use OpenCV's built-in face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            face_inconsistencies = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_region = gray[y:y+h, x:x+w]
                
                # Analyze face region texture
                face_edges = cv2.Canny(face_region, 50, 150)
                edge_density = np.sum(face_edges > 0) / (w * h)
                
                # Check if face region has suspicious characteristics
                # Relax thresholds to reduce false positives on clean photos
                if edge_density < 0.03:  # Too smooth
                    face_inconsistencies.append("Face region too smooth (possible AI generation)")
                elif edge_density > 0.45:  # Too detailed
                    face_inconsistencies.append("Face region has excessive detail (possible artifacts)")
                
                # Check brightness uniformity in face
                face_std = np.std(face_region)
                if face_std < 10:  # Too uniform
                    face_inconsistencies.append("Face lighting too uniform")
            
            return face_inconsistencies, len(faces)
            
        except Exception as e:
            logger.error(f"Error detecting face inconsistencies: {str(e)}")
            return [], 0

    def _extract_exif_signals(self, image_path):
        """Extract simple EXIF signals (camera make/model, datetime)."""
        try:
            with Image.open(image_path) as img:
                exif_data = getattr(img, "_getexif", lambda: None)()
                if not exif_data:
                    return {"has_exif": False}
                # EXIF tags can be numeric; PIL ExifTags to map is optional but avoid import to keep light
                # Common tag IDs
                MAKE = 271
                MODEL = 272
                DATETIME = 306
                SOFTWARE = 305
                make = exif_data.get(MAKE)
                model = exif_data.get(MODEL)
                dt = exif_data.get(DATETIME)
                software = exif_data.get(SOFTWARE)
                return {
                    "has_exif": True,
                    "camera_make": str(make) if make else None,
                    "camera_model": str(model) if model else None,
                    "datetime": str(dt) if dt else None,
                    "software": str(software) if software else None,
                }
        except Exception as e:
            logger.debug(f"EXIF extraction failed: {e}")
            return {"has_exif": False}

    def _ela_score(self, image_path, quality=95):
        """Compute a simple Error Level Analysis score (mean absolute diff). Lower is more uniform."""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                buf = io.BytesIO()
                img.save(buf, 'JPEG', quality=quality)
                buf.seek(0)
                recompressed = Image.open(buf).convert('RGB')
                a = np.asarray(img).astype(np.float32)
                b = np.asarray(recompressed).astype(np.float32)
                diff = np.abs(a - b)
                return float(np.mean(diff))
        except Exception as e:
            logger.debug(f"ELA computation failed: {e}")
            return 0.0

    def _dct_analysis(self, image_path):
        """Advanced DCT coefficient analysis for JPEG tampering detection."""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return {}
            
            # Resize to manageable size for analysis
            if image.shape[0] > 1024 or image.shape[1] > 1024:
                scale = min(1024/image.shape[0], 1024/image.shape[1])
                new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
                image = cv2.resize(image, new_size)
            
            h, w = image.shape
            block_size = 8
            dct_anomalies = []
            dct_variances = []
            
            # Analyze 8x8 DCT blocks (JPEG standard)
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = image[y:y+block_size, x:x+block_size].astype(np.float32)
                    
                    # Apply DCT
                    dct_block = cv2.dct(block)
                    
                    # Check for quantization artifacts
                    # Real JPEG should have specific patterns in DCT coefficients
                    dc_coeff = dct_block[0, 0]  # DC coefficient
                    ac_coeffs = dct_block[1:, 1:].flatten()  # AC coefficients
                    
                    # Variance in AC coefficients
                    ac_var = np.var(ac_coeffs)
                    dct_variances.append(ac_var)
                    
                    # Check for suspicious patterns (too uniform or irregular)
                    zeros_count = np.sum(np.abs(ac_coeffs) < 1e-6)
                    if zeros_count > len(ac_coeffs) * 0.8:  # Too many zeros
                        dct_anomalies.append("excessive_zeros")
                    elif zeros_count < len(ac_coeffs) * 0.1:  # Too few zeros
                        dct_anomalies.append("insufficient_zeros")
            
            return {
                "dct_variance_mean": float(np.mean(dct_variances)),
                "dct_variance_std": float(np.std(dct_variances)),
                "dct_anomaly_ratio": len(dct_anomalies) / max(1, len(dct_variances)),
                "blocks_analyzed": len(dct_variances)
            }
            
        except Exception as e:
            logger.debug(f"DCT analysis failed: {e}")
            return {}

    def _noise_pattern_analysis(self, image_path):
        """Analyze noise patterns that differ between camera sensors and AI generation."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize for consistent analysis
            if gray.shape[0] > 512 or gray.shape[1] > 512:
                scale = min(512/gray.shape[0], 512/gray.shape[1])
                new_size = (int(gray.shape[1]*scale), int(gray.shape[0]*scale))
                gray = cv2.resize(gray, new_size)
            
            # Extract noise using high-pass filter
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray.astype(np.float32) - blur.astype(np.float32)
            
            # Analyze noise characteristics
            noise_std = np.std(noise)
            noise_mean = np.mean(np.abs(noise))
            
            # Check noise distribution (should be roughly Gaussian for camera sensors)
            hist, bins = np.histogram(noise.flatten(), bins=50, range=(-50, 50))
            hist_norm = hist / np.sum(hist)
            
            # Measure deviation from Gaussian
            x = (bins[:-1] + bins[1:]) / 2
            gaussian_approx = np.exp(-0.5 * (x / noise_std)**2)
            gaussian_approx = gaussian_approx / np.sum(gaussian_approx)
            
            kl_divergence = np.sum(hist_norm * np.log(hist_norm / (gaussian_approx + 1e-10) + 1e-10))
            
            # Spatial noise consistency
            h, w = gray.shape
            quad_stds = []
            for i in range(2):
                for j in range(2):
                    quad = noise[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
                    quad_stds.append(np.std(quad))
            
            noise_consistency = np.std(quad_stds) / np.mean(quad_stds) if np.mean(quad_stds) > 0 else 0
            
            return {
                "noise_std": float(noise_std),
                "noise_mean": float(noise_mean),
                "noise_gaussian_deviation": float(kl_divergence),
                "noise_spatial_consistency": float(noise_consistency)
            }
            
        except Exception as e:
            logger.debug(f"Noise analysis failed: {e}")
            return {}

    def _statistical_anomalies(self, image_path):
        """Detect statistical anomalies common in AI-generated images."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {}
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            stats = {}
            
            # Color distribution analysis
            for i, (space, img) in enumerate([("gray", gray), ("hsv_h", hsv[:,:,0]), 
                                            ("hsv_s", hsv[:,:,1]), ("lab_a", lab[:,:,1])]):
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                hist_norm = hist.flatten() / np.sum(hist)
                
                # Calculate entropy
                entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
                stats[f"{space}_entropy"] = float(entropy)
                
                # Peak analysis
                peaks = np.where(hist_norm > np.mean(hist_norm) + 2*np.std(hist_norm))[0]
                stats[f"{space}_peak_count"] = len(peaks)
            
            # Edge gradient analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # AI-generated images often have specific gradient patterns
            stats["gradient_mean"] = float(np.mean(magnitude))
            stats["gradient_std"] = float(np.std(magnitude))
            stats["gradient_skewness"] = float(self._skewness(magnitude.flatten()))
            
            # Pixel value transitions
            diff_h = np.abs(np.diff(gray, axis=1))
            diff_v = np.abs(np.diff(gray, axis=0))
            
            stats["transition_smoothness_h"] = float(np.mean(diff_h))
            stats["transition_smoothness_v"] = float(np.mean(diff_v))
            
            return stats
            
        except Exception as e:
            logger.debug(f"Statistical analysis failed: {e}")
            return {}

    def _skewness(self, data):
        """Calculate skewness of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _metadata_verification(self, image_path):
        """Advanced metadata and format verification."""
        try:
            with Image.open(image_path) as img:
                format_info = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size
                }
                
                # Check for suspicious metadata patterns
                exif = getattr(img, "_getexif", lambda: {})() or {}
                
                # Common AI generation software signatures
                ai_software_signatures = [
                    "midjourney", "dall-e", "stable diffusion", "artificial", 
                    "generated", "synthetic", "ai", "gpt", "chatgpt"
                ]
                
                software = str(exif.get(305, "")).lower()  # Software tag
                description = str(exif.get(270, "")).lower()  # Image description
                
                ai_signature_detected = any(sig in software or sig in description 
                                          for sig in ai_software_signatures)
                
                # Check for missing typical camera metadata
                camera_tags = [271, 272, 306, 33434, 33437]  # Make, Model, DateTime, ExposureTime, FNumber
                present_camera_tags = sum(1 for tag in camera_tags if tag in exif)
                
                # File size vs resolution ratio (AI images often have specific patterns)
                file_size = os.path.getsize(image_path)
                pixel_count = img.size[0] * img.size[1]
                bytes_per_pixel = file_size / pixel_count if pixel_count > 0 else 0
                
                return {
                    "ai_signature_detected": ai_signature_detected,
                    "camera_metadata_completeness": present_camera_tags / len(camera_tags),
                    "bytes_per_pixel": float(bytes_per_pixel),
                    "format": format_info["format"],
                    "suspicious_software": ai_signature_detected
                }
                
        except Exception as e:
            logger.debug(f"Metadata verification failed: {e}")
            return {}

    def _green_indicators(self, features, avg_compression, compression_var, face_issues):
        """Indicators that support natural/unaltered imagery. Returns (green_score, notes)."""
        notes = []
        green = 0.0

        edge_density = features.get('edge_density', 0)
        texture_var = features.get('texture_variance', 0)
        color_uni = features.get('color_uniformity', 0)
        freq_var = features.get('freq_variance', 0)
        b_std = features.get('brightness_std', 0)
        w = max(1, features.get('width', 1))
        h = max(1, features.get('height', 1))
        total_px = w * h

        # Natural range of edges for typical photos
        if 0.04 <= edge_density <= 0.25:
            green += 0.25
            notes.append("Edge density in natural range")

        # Good texture variance suggests non-synthetic content
        if texture_var >= 250:
            green += 0.25
            notes.append("Healthy texture variance")

        # Color variability typical for real images
        if 15 <= color_uni <= 60:
            green += 0.15
            notes.append("Natural color variability")

        # Frequency variance within normal band
        if 1.5 <= freq_var <= 20:
            green += 0.15
            notes.append("Balanced frequency spectrum")

        # Not overly uniform brightness
        if b_std >= 20:
            green += 0.1
            notes.append("Adequate contrast/brightness variation")

        # Compression patterns fairly consistent and not extreme
        if 0.03 <= avg_compression <= 0.20 and compression_var < 0.06:
            green += 0.1
            notes.append("Compression looks consistent with camera/JPEG")

        # Larger images tend to be camera originals; small images are more risky
        if total_px >= (640 * 480):
            green += 0.05
            notes.append("Sufficient resolution")

        # No face issues if faces are detected
        if not face_issues:
            green += 0.05
            notes.append("No facial inconsistencies detected")

        return min(green, 0.8), notes
    
    def detect(self, image_path, session_id=None, progress_dict=None):
        """Main lightweight detection method"""
        try:
            # Update progress
            if progress_dict is not None and session_id is not None:
                progress_dict[session_id] = 20
                emit('progress_update', {'progress': 20, 'message': 'Extracting image features...'}, room=session_id)
            
            # Extract traditional features
            features = self._extract_traditional_features(image_path)
            
            if progress_dict is not None and session_id is not None:
                progress_dict[session_id] = 50
                emit('progress_update', {'progress': 50, 'message': 'Analyzing compression artifacts...'}, room=session_id)
            
            # Analyze compression
            avg_compression, compression_var = self._analyze_compression_artifacts(image_path)
            
            if progress_dict is not None and session_id is not None:
                progress_dict[session_id] = 80
                emit('progress_update', {'progress': 80, 'message': 'Checking for face inconsistencies...'}, room=session_id)
            
            # Face analysis
            face_issues, face_count = self._detect_face_inconsistencies(image_path)

            # EXIF and ELA features
            exif = self._extract_exif_signals(image_path)
            ela = self._ela_score(image_path)
            
            # Advanced forensic analysis
            dct_stats = self._dct_analysis(image_path)
            noise_stats = self._noise_pattern_analysis(image_path)
            statistical_stats = self._statistical_anomalies(image_path)
            metadata_stats = self._metadata_verification(image_path)
            
            # LIGHTWEIGHT DETECTION ALGORITHM (tuned to reduce false positives)
            suspicious_score = 0
            anomalies = []
            
            # Rule 1: Edge density (deepfakes often over-smooth)
            if features.get('edge_density', 0.15) < 0.03:
                suspicious_score += 0.18
                anomalies.append({
                    "type": "Edge smoothing",
                    "severity": "high",
                    "description": "Image appears over-smoothed, typical of AI generation"
                })
            
            # Rule 2: Texture variance (AI often creates unrealistic textures)
            if features.get('texture_variance', 500) < 80:
                suspicious_score += 0.15
                anomalies.append({
                    "type": "Low texture variance",
                    "severity": "medium",
                    "description": "Texture patterns appear artificially uniform"
                })
            
            # Rule 3: Color uniformity (AI struggles with natural color variation)
            if features.get('color_uniformity', 30) < 10:
                suspicious_score += 0.10
                anomalies.append({
                    "type": "Color uniformity",
                    "severity": "medium",
                    "description": "Colors appear artificially uniform"
                })
            
            # Rule 4: Frequency domain analysis
            if features.get('freq_variance', 5) < 1:
                suspicious_score += 0.10
                anomalies.append({
                    "type": "Frequency anomaly",
                    "severity": "medium",
                    "description": "Suspicious frequency patterns detected"
                })
            
            # Rule 5: Compression artifacts
            # Only flag extreme/unusual compression patterns
            if avg_compression > 0.22 or compression_var > 0.12:
                suspicious_score += 0.12
                anomalies.append({
                    "type": "Compression artifacts",
                    "severity": "high",
                    "description": "Unusual compression patterns suggesting manipulation"
                })
            
            # Rule 6: Face inconsistencies
            if face_issues:
                suspicious_score += 0.08 * len(face_issues)
                for issue in face_issues:
                    anomalies.append({
                        "type": "Facial inconsistency",
                        "severity": "high",
                        "description": issue
                    })
            
            # Apply green indicators to offset suspicion when natural cues are strong
            green_score, green_notes = self._green_indicators(features, avg_compression, compression_var, face_issues)

            # Add EXIF-based green signals (light weight to avoid overtrusting metadata)
            if exif.get('has_exif') and (exif.get('camera_make') or exif.get('camera_model')):
                green_score += 0.08
                green_notes.append("Valid EXIF camera make/model present")

            # ELA heuristic: extremely high ELA might indicate local edits; low is neutral/green
            # Calibrate lightly: typical JPEG recompression mean diff often ~1-10 for photo-like images
            if ela < 5.0:
                green_score += 0.05
                green_notes.append("Low ELA difference (uniform compression)")
            elif ela > 25.0:
                suspicious_score += 0.08
                anomalies.append({
                    "type": "ELA anomaly",
                    "severity": "medium",
                    "description": "High error level difference compared to recompression"
                })

            # Advanced forensic checks
            # DCT analysis
            if dct_stats.get("dct_anomaly_ratio", 0) > 0.3:
                suspicious_score += 0.12
                anomalies.append({
                    "type": "DCT anomaly",
                    "severity": "high",
                    "description": "Suspicious DCT coefficient patterns detected"
                })
            elif dct_stats.get("dct_variance_std", 0) > 0 and dct_stats.get("dct_variance_mean", 0) > 100:
                green_score += 0.08
                green_notes.append("Healthy DCT coefficient variation")

            # Noise pattern analysis
            noise_gaussian_dev = noise_stats.get("noise_gaussian_deviation", 0)
            noise_consistency = noise_stats.get("noise_spatial_consistency", 0)
            
            if noise_gaussian_dev > 2.0 or noise_consistency > 0.5:
                suspicious_score += 0.10
                anomalies.append({
                    "type": "Noise anomaly",
                    "severity": "medium",
                    "description": "Unnatural noise patterns detected"
                })
            elif 0.5 <= noise_gaussian_dev <= 1.5 and noise_consistency < 0.3:
                green_score += 0.06
                green_notes.append("Natural sensor noise patterns")

            # Statistical anomalies
            gray_entropy = statistical_stats.get("gray_entropy", 7)
            gradient_skewness = abs(statistical_stats.get("gradient_skewness", 0))
            
            if gray_entropy < 5.5 or gradient_skewness > 2.0:
                suspicious_score += 0.08
                anomalies.append({
                    "type": "Statistical anomaly",
                    "severity": "medium",
                    "description": "Unusual statistical distributions detected"
                })
            elif 6.5 <= gray_entropy <= 7.8 and gradient_skewness < 1.0:
                green_score += 0.05
                green_notes.append("Natural statistical distributions")

            # Metadata verification
            if metadata_stats.get("ai_signature_detected", False):
                suspicious_score += 0.25
                anomalies.append({
                    "type": "AI signature",
                    "severity": "critical",
                    "description": "AI generation software signature detected in metadata"
                })
            
            camera_completeness = metadata_stats.get("camera_metadata_completeness", 0)
            if camera_completeness >= 0.6:
                green_score += 0.10
                green_notes.append("Complete camera metadata present")
            elif camera_completeness == 0:
                suspicious_score += 0.05
                anomalies.append({
                    "type": "Missing metadata",
                    "severity": "low",
                    "description": "No camera metadata present"
                })

            # Bytes per pixel analysis (AI images often have specific compression ratios)
            bytes_per_pixel = metadata_stats.get("bytes_per_pixel", 0)
            if bytes_per_pixel > 0:
                if bytes_per_pixel < 0.5 or bytes_per_pixel > 4.0:
                    suspicious_score += 0.05
                    anomalies.append({
                        "type": "Compression anomaly",
                        "severity": "low",
                        "description": "Unusual file size to resolution ratio"
                    })

            # Face/no-face gating: do not add green facial note if no faces at all
            if face_count == 0:
                green_notes = [n for n in green_notes if n != "No facial inconsistencies detected"]

            # Cap green score and compute adjusted suspicion
            green_score = min(green_score, 0.85)
            adjusted_suspicion = max(0.0, suspicious_score - 0.5 * green_score)
            
            # Try ML classifier if available
            ml_prediction = None
            ml_confidence = None
            detection_method = "Rule-based"
            
            if self.ml_classifier is not None:
                try:
                    feature_vector = self._extract_feature_vector(
                        features, dct_stats, noise_stats, statistical_stats, 
                        metadata_stats, ela, exif
                    )
                    ml_is_fake, ml_fake_prob = self._ml_predict(feature_vector, suspicious_score, green_score)
                    
                    if ml_is_fake is not None:
                        ml_prediction = ml_is_fake
                        ml_confidence = ml_fake_prob
                        detection_method = "ML + Rules"
                        
                        # Use ML prediction as primary, rules as backup
                        is_deepfake = ml_is_fake
                        
                        # Convert ML probability to confidence
                        if ml_is_fake:
                            confidence = float(55 + ml_fake_prob * 40)  # 55-95% range for fakes
                        else:
                            confidence = float(95 - ml_fake_prob * 40)  # 55-95% range for real
                        
                except Exception as e:
                    logger.warning(f"ML classifier error, falling back to rules: {e}")
            
            # Fallback to rule-based detection if ML not available
            if ml_prediction is None:
                # Threshold configurable via environment (Config)
                threshold = getattr(Config, 'IMAGE_FAKE_THRESHOLD', 0.8)
                try:
                    threshold = float(threshold)
                except Exception:
                    threshold = 0.8
                is_deepfake = adjusted_suspicion > threshold
                
                # Calculate confidence (higher score = more confident it's fake)
                if is_deepfake:
                    confidence = float(min(55 + adjusted_suspicion * 45, 92))
                else:
                    # Confidence for real increases with green_score and low suspicion
                    confidence = float(min(90, 60 + 25 * green_score - 10 * adjusted_suspicion))
            
            if progress_dict is not None and session_id is not None:
                progress_dict[session_id] = 100
                emit('progress_update', {'progress': 100, 'message': 'Analysis complete'}, room=session_id)
            
            # Optional external verification
            external_verification = {}
            try:
                from app.utils.external_verification import verify_with_external_services
                external_verification = verify_with_external_services(image_path, {
                    "isDeepfake": is_deepfake,
                    "confidence": confidence
                })
                
                # Apply confidence boost from external services
                if external_verification.get("confidence_boost", 0) != 0:
                    confidence = min(95, max(5, confidence + external_verification["confidence_boost"] * 100))
                    
            except Exception as e:
                logger.debug(f"External verification failed: {e}")
                external_verification = {"enabled": False, "error": str(e)}
            
            return {
                "isDeepfake": bool(is_deepfake),
                "confidence": float(confidence),
                "processingTime": 0.5,  # Much faster!
                "anomalies": anomalies + ([{"type": "Green indicator", "severity": "info", "description": note} for note in green_notes] if green_notes else []),
                "mediaType": "image",
                "timestamp": datetime.now().isoformat(),
                "modelUsed": f"Lightweight CV Analysis v1.1 (tuned) - {detection_method}",
                "features": {
                    "edge_density": features.get('edge_density', 0),
                    "texture_variance": features.get('texture_variance', 0),
                    "color_uniformity": features.get('color_uniformity', 0),
                    "freq_variance": features.get('freq_variance', 0),
                    "avg_compression": float(avg_compression),
                    "compression_var": float(compression_var),
                        "ela_score": float(ela),
                        "exif_present": bool(exif.get('has_exif', False)),
                        "dct_anomaly_ratio": dct_stats.get("dct_anomaly_ratio", 0),
                        "noise_gaussian_deviation": noise_stats.get("noise_gaussian_deviation", 0),
                        "noise_spatial_consistency": noise_stats.get("noise_spatial_consistency", 0),
                        "gray_entropy": statistical_stats.get("gray_entropy", 0),
                        "gradient_skewness": statistical_stats.get("gradient_skewness", 0),
                        "ai_signature_detected": metadata_stats.get("ai_signature_detected", False),
                        "camera_metadata_completeness": metadata_stats.get("camera_metadata_completeness", 0),
                        "bytes_per_pixel": metadata_stats.get("bytes_per_pixel", 0),
                        "suspicious_score_raw": float(suspicious_score),
                        "green_score": float(green_score),
                        "suspicious_score": float(adjusted_suspicion),
                        "ml_prediction": ml_prediction,
                        "ml_confidence": ml_confidence,
                        "external_verification": external_verification
                }
            }
            
        except Exception as e:
            logger.error(f"Error in lightweight detection: {str(e)}")
            return {
                "isDeepfake": False,
                "confidence": 50.0,
                "processingTime": 0.1,
                "anomalies": [],
                "mediaType": "image",
                "timestamp": datetime.now().isoformat(),
                "modelUsed": "Error",
                "error": str(e)
            }

# Global detector instance
detector = LightweightImageDetector()

def analyze_image(file_path, session_id=None, progress_dict=None):
    """Main function called by the Flask app"""
    return detector.detect(file_path, session_id, progress_dict)