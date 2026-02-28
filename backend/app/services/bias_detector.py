"""
Bias Detector - Detects potential biases in retrieved sources
"""
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BiasDetector:
    """
    Detects potential biases in source documents
    Flags issues like industry sponsorship, recency bias, etc.
    """
    
    def __init__(self):
        self.industry_keywords = [
            "sponsored by",
            "funded by",
            "pharmaceutical company",
            "industry-sponsored",
            "commercial sponsor",
            "corporate funding",
        ]
    
    def analyze_sources(
        self,
        documents: List[Dict[str, Any]],
        query: str = None
    ) -> Dict[str, Any]:
        """
        Analyze sources for potential biases
        
        Args:
            documents: List of retrieved documents with metadata
            query: Original query (optional)
            
        Returns:
            Bias analysis report
        """
        if not documents:
            return {
                "has_bias": False,
                "bias_flags": [],
                "bias_score": 0.0,
                "recommendations": []
            }
        
        bias_flags = []
        
        # Check 1: Industry sponsorship bias
        industry_sponsored = self._check_industry_sponsorship(documents)
        if industry_sponsored["ratio"] > 0.7:
            bias_flags.append({
                "type": "industry_sponsorship",
                "severity": "high",
                "message": f"{industry_sponsored['ratio']:.0%} of sources are industry-sponsored",
                "details": industry_sponsored
            })
        elif industry_sponsored["ratio"] > 0.5:
            bias_flags.append({
                "type": "industry_sponsorship",
                "severity": "medium",
                "message": f"{industry_sponsored['ratio']:.0%} of sources are industry-sponsored",
                "details": industry_sponsored
            })
        
        # Check 2: Recency bias
        recency_bias = self._check_recency_bias(documents)
        if recency_bias["all_recent"]:
            bias_flags.append({
                "type": "recency_bias",
                "severity": "medium",
                "message": "All sources are from 2020 or later",
                "details": recency_bias
            })
        
        # Check 3: Source diversity
        source_diversity = self._check_source_diversity(documents)
        if source_diversity["diversity_score"] < 0.3:
            bias_flags.append({
                "type": "low_diversity",
                "severity": "medium",
                "message": "Limited diversity in source types",
                "details": source_diversity
            })
        
        # Check 4: Geographic bias
        geographic_bias = self._check_geographic_bias(documents)
        if geographic_bias["single_region_ratio"] > 0.8:
            bias_flags.append({
                "type": "geographic_bias",
                "severity": "low",
                "message": f"Most sources from {geographic_bias['dominant_region']}",
                "details": geographic_bias
            })
        
        # Calculate overall bias score
        bias_score = self._calculate_bias_score(bias_flags)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(bias_flags)
        
        return {
            "has_bias": len(bias_flags) > 0,
            "bias_flags": bias_flags,
            "bias_score": bias_score,
            "recommendations": recommendations,
            "analysis": {
                "industry_sponsorship": industry_sponsored,
                "recency_bias": recency_bias,
                "source_diversity": source_diversity,
                "geographic_bias": geographic_bias,
            }
        }
    
    def _check_industry_sponsorship(self, documents: List[Dict]) -> Dict:
        """Check for industry sponsorship bias"""
        total = len(documents)
        industry_count = 0
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            
            # Check industry_sponsored flag
            if metadata.get("industry_sponsored", False):
                industry_count += 1
                continue
            
            # Check text for industry keywords
            text = doc.get("text", "").lower()
            title = metadata.get("title", "").lower()
            
            for keyword in self.industry_keywords:
                if keyword in text or keyword in title:
                    industry_count += 1
                    break
        
        return {
            "total_sources": total,
            "industry_sponsored": industry_count,
            "ratio": industry_count / total if total > 0 else 0.0
        }
    
    def _check_recency_bias(self, documents: List[Dict]) -> Dict:
        """Check for recency bias (all sources too recent)"""
        total = len(documents)
        recent_count = 0
        years = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            pub_date = metadata.get("publication_date")
            
            if pub_date:
                try:
                    if isinstance(pub_date, str):
                        year = int(pub_date[:4])
                    else:
                        year = pub_date.year
                    
                    years.append(year)
                    if year >= 2020:
                        recent_count += 1
                except:
                    pass
        
        return {
            "total_sources": total,
            "recent_sources": recent_count,
            "all_recent": recent_count == total and total > 0,
            "year_range": f"{min(years)}-{max(years)}" if years else "unknown",
            "median_year": sorted(years)[len(years)//2] if years else None
        }
    
    def _check_source_diversity(self, documents: List[Dict]) -> Dict:
        """Check diversity of source types"""
        source_types = {}
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            source_type = metadata.get("source_type", "unknown")
            source_types[source_type] = source_types.get(source_type, 0) + 1
        
        total = len(documents)
        unique_types = len(source_types)
        
        # Calculate diversity score (0-1, higher is more diverse)
        if total == 0:
            diversity_score = 0.0
        else:
            # Shannon entropy-based diversity
            import math
            entropy = 0
            for count in source_types.values():
                p = count / total
                entropy -= p * math.log2(p)
            
            # Normalize by max possible entropy
            max_entropy = math.log2(total) if total > 1 else 1
            diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            "total_sources": total,
            "unique_types": unique_types,
            "source_distribution": source_types,
            "diversity_score": diversity_score
        }
    
    def _check_geographic_bias(self, documents: List[Dict]) -> Dict:
        """Check for geographic bias"""
        regions = {}
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            region = metadata.get("region", "unknown")
            regions[region] = regions.get(region, 0) + 1
        
        total = len(documents)
        
        if not regions or total == 0:
            return {
                "total_sources": total,
                "regions": regions,
                "dominant_region": "unknown",
                "single_region_ratio": 0.0
            }
        
        dominant_region = max(regions, key=regions.get)
        dominant_count = regions[dominant_region]
        
        return {
            "total_sources": total,
            "regions": regions,
            "dominant_region": dominant_region,
            "single_region_ratio": dominant_count / total
        }
    
    def _calculate_bias_score(self, bias_flags: List[Dict]) -> float:
        """
        Calculate overall bias score (0-1, higher is more biased)
        """
        if not bias_flags:
            return 0.0
        
        severity_weights = {
            "high": 1.0,
            "medium": 0.6,
            "low": 0.3
        }
        
        total_weight = sum(
            severity_weights.get(flag["severity"], 0.5)
            for flag in bias_flags
        )
        
        # Normalize to 0-1 scale
        max_possible = len(bias_flags) * 1.0
        return min(1.0, total_weight / max_possible) if max_possible > 0 else 0.0
    
    def _generate_recommendations(self, bias_flags: List[Dict]) -> List[str]:
        """Generate recommendations based on detected biases"""
        recommendations = []
        
        for flag in bias_flags:
            if flag["type"] == "industry_sponsorship":
                recommendations.append(
                    "Consider seeking additional independent research sources"
                )
                recommendations.append(
                    "Review conflict of interest statements in cited papers"
                )
            
            elif flag["type"] == "recency_bias":
                recommendations.append(
                    "Include historical context and foundational research"
                )
                recommendations.append(
                    "Consider long-term safety and efficacy data"
                )
            
            elif flag["type"] == "low_diversity":
                recommendations.append(
                    "Expand search to include diverse source types"
                )
                recommendations.append(
                    "Consider meta-analyses and systematic reviews"
                )
            
            elif flag["type"] == "geographic_bias":
                recommendations.append(
                    "Consider research from multiple geographic regions"
                )
                recommendations.append(
                    "Be aware of population-specific findings"
                )
        
        return list(set(recommendations))  # Remove duplicates


# Singleton instance
_bias_detector = None


def get_bias_detector() -> BiasDetector:
    """Get or create the bias detector singleton"""
    global _bias_detector
    if _bias_detector is None:
        _bias_detector = BiasDetector()
    return _bias_detector
