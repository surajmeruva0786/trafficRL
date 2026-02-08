"""
Level of Service (LOS) Calculator Module

Implements Highway Capacity Manual (HCM) 2010 Level of Service classification
for signalized intersections based on control delay.
"""

from typing import Dict, Tuple
from enum import Enum


class LOSGrade(Enum):
    """Level of Service grade enumeration."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class LevelOfServiceCalculator:
    """
    Calculate Level of Service (LOS) classification for traffic facilities.
    
    LOS is a qualitative measure describing operational conditions within
    a traffic stream, based on service measures such as speed, travel time,
    freedom to maneuver, traffic interruptions, and comfort.
    
    This implementation uses HCM 2010 criteria for signalized intersections,
    which is based on control delay per vehicle.
    """
    
    # HCM 2010 LOS thresholds for signalized intersections (seconds of delay)
    LOS_THRESHOLDS = {
        LOSGrade.A: (0, 10),      # ≤ 10 seconds
        LOSGrade.B: (10, 20),     # > 10 and ≤ 20 seconds
        LOSGrade.C: (20, 35),     # > 20 and ≤ 35 seconds
        LOSGrade.D: (35, 55),     # > 35 and ≤ 55 seconds
        LOSGrade.E: (55, 80),     # > 55 and ≤ 80 seconds
        LOSGrade.F: (80, float('inf'))  # > 80 seconds
    }
    
    # Descriptions of each LOS grade
    LOS_DESCRIPTIONS = {
        LOSGrade.A: "Free flow - Minimal delays, excellent operation",
        LOSGrade.B: "Stable flow - Slight delays, very good operation",
        LOSGrade.C: "Stable flow - Acceptable delays, good operation",
        LOSGrade.D: "Approaching unstable - Tolerable delays, satisfactory operation",
        LOSGrade.E: "Unstable flow - Significant delays, poor operation",
        LOSGrade.F: "Forced flow - Excessive delays, unacceptable operation"
    }
    
    # Detailed operational characteristics
    LOS_CHARACTERISTICS = {
        LOSGrade.A: {
            'description': 'Free flow',
            'delay_range': '≤ 10 seconds',
            'operations': 'Progression is extremely favorable, and most vehicles arrive during the green phase. Most vehicles do not stop at all.',
            'driver_comfort': 'Excellent',
            'maneuverability': 'Excellent'
        },
        LOSGrade.B: {
            'description': 'Stable flow',
            'delay_range': '> 10 to 20 seconds',
            'operations': 'Progression is highly favorable, and many vehicles arrive during the green phase. Some vehicles stop.',
            'driver_comfort': 'Very good',
            'maneuverability': 'Very good'
        },
        LOSGrade.C: {
            'description': 'Stable flow',
            'delay_range': '> 20 to 35 seconds',
            'operations': 'Higher delays result from fair progression and/or longer cycle lengths. Individual cycle failures may begin to appear.',
            'driver_comfort': 'Good',
            'maneuverability': 'Good'
        },
        LOSGrade.D: {
            'description': 'Approaching unstable',
            'delay_range': '> 35 to 55 seconds',
            'operations': 'Noticeable congestion. Longer delays result from unfavorable progression, long cycle lengths, or high v/c ratios.',
            'driver_comfort': 'Satisfactory',
            'maneuverability': 'Fair'
        },
        LOSGrade.E: {
            'description': 'Unstable flow',
            'delay_range': '> 55 to 80 seconds',
            'operations': 'High delays indicate poor progression, long cycle lengths, and high v/c ratios. Individual cycle failures are frequent.',
            'driver_comfort': 'Poor',
            'maneuverability': 'Poor'
        },
        LOSGrade.F: {
            'description': 'Forced flow',
            'delay_range': '> 80 seconds',
            'operations': 'Unacceptable delays. Represents jammed conditions. Occurs with oversaturation, when arrival flow rates exceed capacity.',
            'driver_comfort': 'Unacceptable',
            'maneuverability': 'Very poor'
        }
    }
    
    @staticmethod
    def get_los_from_delay(delay_per_vehicle: float) -> LOSGrade:
        """
        Determine LOS grade based on average control delay per vehicle.
        
        Args:
            delay_per_vehicle: Average control delay in seconds per vehicle
            
        Returns:
            LOSGrade enum value (A through F)
        """
        for grade, (lower, upper) in LevelOfServiceCalculator.LOS_THRESHOLDS.items():
            if lower < delay_per_vehicle <= upper:
                return grade
        
        # If delay is exactly 0 or negative, return A
        if delay_per_vehicle <= 0:
            return LOSGrade.A
        
        # Fallback to F for any edge cases
        return LOSGrade.F
    
    @staticmethod
    def get_los_description(los: LOSGrade) -> str:
        """
        Get brief description of LOS grade.
        
        Args:
            los: LOSGrade enum value
            
        Returns:
            Brief description string
        """
        return LevelOfServiceCalculator.LOS_DESCRIPTIONS.get(
            los, "Unknown LOS"
        )
    
    @staticmethod
    def get_los_characteristics(los: LOSGrade) -> Dict[str, str]:
        """
        Get detailed operational characteristics for LOS grade.
        
        Args:
            los: LOSGrade enum value
            
        Returns:
            Dictionary with detailed characteristics
        """
        return LevelOfServiceCalculator.LOS_CHARACTERISTICS.get(
            los, {}
        )
    
    @staticmethod
    def get_delay_threshold(los: LOSGrade) -> Tuple[float, float]:
        """
        Get delay threshold range for a given LOS grade.
        
        Args:
            los: LOSGrade enum value
            
        Returns:
            Tuple of (lower_bound, upper_bound) in seconds
        """
        return LevelOfServiceCalculator.LOS_THRESHOLDS.get(
            los, (0, 0)
        )
    
    @staticmethod
    def get_all_thresholds() -> Dict[LOSGrade, Tuple[float, float]]:
        """
        Get all LOS thresholds.
        
        Returns:
            Dictionary mapping LOS grades to delay thresholds
        """
        return LevelOfServiceCalculator.LOS_THRESHOLDS.copy()
    
    @staticmethod
    def calculate_los_distribution(delays: list) -> Dict[LOSGrade, Dict[str, float]]:
        """
        Calculate distribution of LOS grades across multiple delay measurements.
        
        Args:
            delays: List of delay values in seconds
            
        Returns:
            Dictionary mapping LOS grades to count and percentage
        """
        if not delays:
            return {}
        
        los_counts = {grade: 0 for grade in LOSGrade}
        
        for delay in delays:
            los = LevelOfServiceCalculator.get_los_from_delay(delay)
            los_counts[los] += 1
        
        total = len(delays)
        distribution = {}
        
        for grade, count in los_counts.items():
            if count > 0:
                distribution[grade] = {
                    'count': count,
                    'percentage': (count / total) * 100,
                    'description': LevelOfServiceCalculator.get_los_description(grade)
                }
        
        return distribution
    
    @staticmethod
    def get_los_score(los: LOSGrade) -> float:
        """
        Convert LOS grade to numerical score for optimization.
        
        A = 6.0 (best)
        B = 5.0
        C = 4.0
        D = 3.0
        E = 2.0
        F = 1.0 (worst)
        
        Args:
            los: LOSGrade enum value
            
        Returns:
            Numerical score (1.0 to 6.0)
        """
        score_map = {
            LOSGrade.A: 6.0,
            LOSGrade.B: 5.0,
            LOSGrade.C: 4.0,
            LOSGrade.D: 3.0,
            LOSGrade.E: 2.0,
            LOSGrade.F: 1.0
        }
        return score_map.get(los, 1.0)
    
    @staticmethod
    def format_los_report(delay: float) -> str:
        """
        Generate formatted LOS report for a given delay.
        
        Args:
            delay: Average delay in seconds
            
        Returns:
            Formatted string report
        """
        los = LevelOfServiceCalculator.get_los_from_delay(delay)
        chars = LevelOfServiceCalculator.get_los_characteristics(los)
        
        report = f"""
Level of Service Analysis
==========================
Average Delay: {delay:.2f} seconds
LOS Grade: {los.value}
Description: {chars.get('description', 'N/A')}
Delay Range: {chars.get('delay_range', 'N/A')}

Operational Characteristics:
{chars.get('operations', 'N/A')}

Driver Comfort: {chars.get('driver_comfort', 'N/A')}
Maneuverability: {chars.get('maneuverability', 'N/A')}
        """.strip()
        
        return report
    
    @staticmethod
    def is_acceptable_los(los: LOSGrade, threshold: LOSGrade = LOSGrade.D) -> bool:
        """
        Check if LOS meets acceptability threshold.
        
        Typically, LOS D or better is considered acceptable for urban areas.
        
        Args:
            los: LOSGrade to check
            threshold: Minimum acceptable LOS (default: D)
            
        Returns:
            True if LOS is acceptable, False otherwise
        """
        score = LevelOfServiceCalculator.get_los_score(los)
        threshold_score = LevelOfServiceCalculator.get_los_score(threshold)
        return score >= threshold_score
    
    def __repr__(self) -> str:
        return "LevelOfServiceCalculator(HCM 2010 criteria)"
