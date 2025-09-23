#!/usr/bin/env python3
"""
External Service Integration for Enhanced Image Verification

This module provides integration with external APIs and services for:
- Reverse image search
- Known AI model signature detection
- Cross-validation with other detection services
- Blockchain verification (if implemented)
"""

import os
import requests
import hashlib
import time
import json
from datetime import datetime
import logging
from config.settings import Config

logger = logging.getLogger(__name__)

class ExternalVerificationService:
    def __init__(self):
        self.google_api_key = os.environ.get('GOOGLE_CUSTOM_SEARCH_API_KEY')
        self.google_cx = os.environ.get('GOOGLE_CUSTOM_SEARCH_CX')
        self.tineye_api_key = os.environ.get('TINEYE_API_KEY')
        self.hive_api_key = os.environ.get('HIVE_AI_API_KEY')
        
    def reverse_image_search(self, image_path, max_results=5):
        """Perform reverse image search to find similar images."""
        results = {
            "found_matches": False,
            "match_count": 0,
            "earliest_date": None,
            "source_domains": [],
            "suspicious_patterns": []
        }
        
        try:
            # Try Google Custom Search if configured
            if self.google_api_key and self.google_cx:
                google_results = self._google_reverse_search(image_path, max_results)
                if google_results:
                    results.update(google_results)
            
            # Try TinEye if configured
            elif self.tineye_api_key:
                tineye_results = self._tineye_search(image_path, max_results)
                if tineye_results:
                    results.update(tineye_results)
            
            # Analyze results for suspicious patterns
            self._analyze_search_patterns(results)
            
        except Exception as e:
            logger.warning(f"Reverse image search failed: {e}")
            
        return results
    
    def _google_reverse_search(self, image_path, max_results):
        """Google Custom Search reverse image search."""
        try:
            # Note: This is a simplified example. Google's actual reverse image search
            # API is more complex and may require different endpoints
            
            # Upload image and get search results (placeholder implementation)
            # In practice, you'd need to upload the image and use the search API
            
            # For now, return mock data structure
            return {
                "found_matches": False,
                "match_count": 0,
                "service_used": "Google Custom Search"
            }
            
        except Exception as e:
            logger.debug(f"Google search failed: {e}")
            return None
    
    def _tineye_search(self, image_path, max_results):
        """TinEye reverse image search."""
        try:
            # TinEye API implementation
            # This would require proper TinEye API integration
            
            return {
                "found_matches": False,
                "match_count": 0,
                "service_used": "TinEye"
            }
            
        except Exception as e:
            logger.debug(f"TinEye search failed: {e}")
            return None
    
    def _analyze_search_patterns(self, results):
        """Analyze reverse search results for suspicious patterns."""
        if results.get("match_count", 0) > 0:
            # Check for AI generation sites
            ai_domains = [
                "midjourney.com", "openai.com", "stability.ai", "lexica.art",
                "artstation.com", "deviantart.com", "aiart", "generated"
            ]
            
            for domain in results.get("source_domains", []):
                if any(ai_site in domain.lower() for ai_site in ai_domains):
                    results["suspicious_patterns"].append(f"Found on AI art platform: {domain}")
            
            # Check for very recent first appearance (might indicate fresh generation)
            earliest = results.get("earliest_date")
            if earliest:
                try:
                    earliest_dt = datetime.fromisoformat(earliest)
                    days_ago = (datetime.now() - earliest_dt).days
                    if days_ago < 7:
                        results["suspicious_patterns"].append(f"Very recent first appearance: {days_ago} days ago")
                except:
                    pass
    
    def hive_ai_verification(self, image_path):
        """Use Hive AI moderation API for additional verification."""
        if not self.hive_ai_api_key:
            return {"available": False, "reason": "API key not configured"}
        
        try:
            # Hive AI API call (placeholder - requires actual implementation)
            # This would upload the image to Hive AI and get their classification
            
            return {
                "available": True,
                "is_generated": False,
                "confidence": 0.5,
                "service": "Hive AI",
                "details": "Placeholder implementation"
            }
            
        except Exception as e:
            logger.warning(f"Hive AI verification failed: {e}")
            return {"available": False, "error": str(e)}
    
    def compute_perceptual_hash(self, image_path):
        """Compute perceptual hash for duplicate detection."""
        try:
            from PIL import Image
            import imagehash
            
            with Image.open(image_path) as img:
                # Different hash algorithms for robustness
                hashes = {
                    "average_hash": str(imagehash.average_hash(img)),
                    "perceptual_hash": str(imagehash.phash(img)),
                    "difference_hash": str(imagehash.dhash(img)),
                    "wavelet_hash": str(imagehash.whash(img))
                }
                
            return hashes
            
        except ImportError:
            logger.warning("imagehash library not installed, skipping perceptual hashing")
            return {}
        except Exception as e:
            logger.debug(f"Perceptual hashing failed: {e}")
            return {}
    
    def blockchain_verification(self, image_path):
        """Verify image against blockchain records (if implemented)."""
        try:
            # Compute file hash
            with open(image_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Check against blockchain registry (placeholder)
            # In a real implementation, this would query your blockchain contract
            
            return {
                "file_hash": file_hash,
                "blockchain_verified": False,
                "registration_date": None,
                "verification_available": False
            }
            
        except Exception as e:
            logger.debug(f"Blockchain verification failed: {e}")
            return {"verification_available": False, "error": str(e)}
    
    def cross_validate_detection(self, image_path, primary_result):
        """Cross-validate with multiple external services."""
        validation_results = {
            "external_services_checked": 0,
            "consensus_fake": 0,
            "consensus_real": 0,
            "confidence_boost": 0.0,
            "services": []
        }
        
        try:
            # Reverse image search
            reverse_search = self.reverse_image_search(image_path)
            if reverse_search.get("found_matches"):
                validation_results["external_services_checked"] += 1
                validation_results["services"].append({
                    "name": "Reverse Image Search",
                    "result": "suspicious" if reverse_search.get("suspicious_patterns") else "clean",
                    "details": reverse_search
                })
                
                if reverse_search.get("suspicious_patterns"):
                    validation_results["consensus_fake"] += 1
                else:
                    validation_results["consensus_real"] += 1
            
            # Hive AI verification
            hive_result = self.hive_ai_verification(image_path)
            if hive_result.get("available"):
                validation_results["external_services_checked"] += 1
                is_fake = hive_result.get("is_generated", False)
                validation_results["services"].append({
                    "name": "Hive AI",
                    "result": "fake" if is_fake else "real",
                    "confidence": hive_result.get("confidence", 0.5),
                    "details": hive_result
                })
                
                if is_fake:
                    validation_results["consensus_fake"] += 1
                else:
                    validation_results["consensus_real"] += 1
            
            # Calculate confidence boost based on consensus
            if validation_results["external_services_checked"] > 0:
                fake_ratio = validation_results["consensus_fake"] / validation_results["external_services_checked"]
                real_ratio = validation_results["consensus_real"] / validation_results["external_services_checked"]
                
                # Boost confidence if external services agree with primary result
                primary_is_fake = primary_result.get("isDeepfake", False)
                
                if primary_is_fake and fake_ratio > 0.5:
                    validation_results["confidence_boost"] = 0.1 * fake_ratio
                elif not primary_is_fake and real_ratio > 0.5:
                    validation_results["confidence_boost"] = 0.1 * real_ratio
                else:
                    # External services disagree with primary result
                    validation_results["confidence_boost"] = -0.05
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            validation_results["error"] = str(e)
        
        return validation_results

# Global service instance
external_verification_service = ExternalVerificationService()

def verify_with_external_services(image_path, primary_result):
    """Main function to verify image with external services."""
    if not getattr(Config, 'EXTERNAL_VERIFICATION_ENABLED', 'false').lower() == 'true':
        return {"enabled": False}
    
    return external_verification_service.cross_validate_detection(image_path, primary_result)