# ==========================================
# ROAD INFORMATION DATABASE
# ==========================================

HIGHWAYS = {

    "NH 44": {
        "name": "National Highway 44",
        "length": "Approximately 3,745 km",
        "start": "Srinagar, Jammu & Kashmir",
        "end": "Kanyakumari, Tamil Nadu",

        "states": [
            "Jammu & Kashmir",
            "Punjab",
            "Haryana",
            "Delhi",
            "Uttar Pradesh",
            "Rajasthan",
            "Madhya Pradesh",
            "Maharashtra",
            "Telangana",
            "Andhra Pradesh",
            "Karnataka",
            "Tamil Nadu"
        ],

        "major_cities": [
            "Srinagar",
            "Jammu",
            "Ludhiana",
            "Delhi",
            "Agra",
            "Gwalior",
            "Nagpur",
            "Hyderabad",
            "Bengaluru",
            "Salem",
            "Madurai",
            "Kanyakumari"
        ],

        "description":
            "India's longest National Highway and a major north-south "
            "transport corridor connecting Srinagar with Kanyakumari.",

        "importance":
            "Major north-south highway connecting multiple regions of India."
    },


    "NH 48": {
        "name": "National Highway 48",
        "length": "Approximately 2,807 km",
        "start": "Delhi",
        "end": "Chennai, Tamil Nadu",

        "states": [
            "Delhi",
            "Haryana",
            "Rajasthan",
            "Gujarat",
            "Maharashtra",
            "Karnataka",
            "Tamil Nadu"
        ],

        "major_cities": [
            "Delhi",
            "Gurugram",
            "Jaipur",
            "Udaipur",
            "Ahmedabad",
            "Mumbai",
            "Pune",
            "Bengaluru",
            "Chennai"
        ],

        "description":
            "A major northwestern and southern corridor connecting "
            "Delhi with Chennai.",

        "importance":
            "Important route for passenger and freight transportation."
    },


    "NH 16": {
        "name": "National Highway 16",
        "length": "Approximately 1,711 km",
        "start": "Kolkata, West Bengal",
        "end": "Chennai, Tamil Nadu",

        "states": [
            "West Bengal",
            "Odisha",
            "Andhra Pradesh",
            "Tamil Nadu"
        ],

        "major_cities": [
            "Kolkata",
            "Bhubaneswar",
            "Cuttack",
            "Visakhapatnam",
            "Vijayawada",
            "Nellore",
            "Chennai"
        ],

        "description":
            "A major east-coast highway running through eastern and "
            "southeastern India.",

        "importance":
            "Important coastal transport corridor."
    },


    "NH 27": {
        "name": "National Highway 27",
        "length": "Approximately 3,507 km",
        "start": "Porbandar, Gujarat",
        "end": "Silchar, Assam",

        "states": [
            "Gujarat",
            "Rajasthan",
            "Madhya Pradesh",
            "Uttar Pradesh",
            "Bihar",
            "West Bengal",
            "Assam"
        ],

        "major_cities": [
            "Porbandar",
            "Udaipur",
            "Kota",
            "Lucknow",
            "Gorakhpur",
            "Muzaffarpur",
            "Siliguri",
            "Guwahati",
            "Silchar"
        ],

        "description":
            "A major east-west highway connecting western India "
            "with northeastern India.",

        "importance":
            "Provides an important cross-country transportation route."
    },


    "NH 19": {
        "name": "National Highway 19",
        "length": "Approximately 1,435 km",
        "start": "Delhi",
        "end": "Kolkata, West Bengal",

        "states": [
            "Delhi",
            "Haryana",
            "Uttar Pradesh",
            "Bihar",
            "Jharkhand",
            "West Bengal"
        ],

        "major_cities": [
            "Delhi",
            "Agra",
            "Kanpur",
            "Prayagraj",
            "Varanasi",
            "Dhanbad",
            "Asansol",
            "Kolkata"
        ],

        "description":
            "A major highway connecting Delhi with Kolkata through "
            "the northern and eastern parts of India.",

        "importance":
            "Important freight and passenger transportation corridor."
    },


    "NH 66": {
        "name": "National Highway 66",
        "length": "Approximately 1,622 km",
        "start": "Panvel, Maharashtra",
        "end": "Kanyakumari, Tamil Nadu",

        "states": [
            "Maharashtra",
            "Goa",
            "Karnataka",
            "Kerala",
            "Tamil Nadu"
        ],

        "major_cities": [
            "Mumbai",
            "Panaji",
            "Mangaluru",
            "Kozhikode",
            "Kochi",
            "Thiruvananthapuram",
            "Kanyakumari"
        ],

        "description":
            "A major coastal highway running along India's western "
            "coast before reaching Tamil Nadu.",

        "importance":
            "Important coastal route connecting western and southern India."
    },


    "NH 65": {
        "name": "National Highway 65",
        "length": "Approximately 962 km",
        "start": "Pune, Maharashtra",
        "end": "Machilipatnam, Andhra Pradesh",

        "states": [
            "Maharashtra",
            "Karnataka",
            "Telangana",
            "Andhra Pradesh"
        ],

        "major_cities": [
            "Pune",
            "Solapur",
            "Kalaburagi",
            "Hyderabad",
            "Suryapet",
            "Vijayawada",
            "Machilipatnam"
        ],

        "description":
            "An important east-west highway connecting Maharashtra "
            "with Andhra Pradesh.",

        "importance":
            "Important transportation corridor across central and southern India."
    }
}


# ==========================================
# SEARCH HIGHWAY
# ==========================================

def get_highway_info(highway_name):
    """
    Return information about a highway.
    """

    if not highway_name:
        return None

    highway_name = highway_name.strip().upper()

    return HIGHWAYS.get(highway_name)