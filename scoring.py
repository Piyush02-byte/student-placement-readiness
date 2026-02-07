"""
Scoring utilities for placement readiness assessment
"""

def readiness_level(score: float) -> str:
    """
    Categorize readiness score into levels
    
    Args:
        score: Readiness score (0-100)
        
    Returns:
        Readiness level: 'Low', 'Medium', or 'High'
    """
    if score < 60:
        return "Low"
    elif score < 80:
        return "Medium"
    else:
        return "High"


def get_level_description(level: str) -> str:
    """Get detailed description for readiness level"""
    descriptions = {
        "Low": "Significant improvement needed across multiple areas",
        "Medium": "Good foundation with room for targeted improvements",
        "High": "Strong profile, well-prepared for placement opportunities"
    }
    return descriptions.get(level, "Unknown level")