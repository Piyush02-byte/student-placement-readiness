def calculate_readiness(row):
    cgpa_score = (row["CGPA"] / 10) * 100
    coding_score = (row["Coding_Skills"] / 10) * 100
    comm_score = (row["Communication_Skills"] / 10) * 100
    projects_score = min(row["Projects"], 5) / 5 * 100
    internship_score = min(row["Internships"], 3) / 3 * 100

    readiness = (
        0.30 * cgpa_score +
        0.25 * coding_score +
        0.20 * comm_score +
        0.10 * projects_score +
        0.05 * internship_score
    )
    return readiness


def readiness_level(score):
    if score <= 40:
        return "Low"
    elif score <= 70:
        return "Medium"
    else:
        return "High"
