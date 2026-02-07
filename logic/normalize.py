import re

def normalize_team(name: str) -> str:
    if not name:
        return ""

    n = name.lower().strip()

    # remove punctuation
    n = re.sub(r"[^\w\s]", "", n)

    # common abbreviations (whole-word "st"/"st.")
    n = re.sub(r"\bst\.?\b", "state", n)

    # remove mascots / trailing words that sportsbooks add
    remove_words = {
        "knights", "scarlet", "bruins", "demons", "blue", "red",
        "wildcats", "panthers", "bulldogs", "tigers", "bears",
        "hawks", "eagles", "wolves", "wolfpack", "pack",
        "trojans", "rebels", "aztecs", "hoosiers", "cowboys",
        "crimson", "tide", "mountaineers", "sun", "devils", "cougars",
        "golden", "buffaloes", "bluejays", "titans", "dukes", "owls",
        "patriots", "hoyas", "yellow", "jackets", "revolutionaries",
        "fighting", "illini", "hawkeyes", "cardinals", "lions",
        "spartans", "gophers", "lobos", "mean", "green", "irish",
        "sooners", "beavers", "waves", "pilots", "friars", "gaels",
        "toreros", "dons", "broncos", "redhawks", "pirates", "jaguars",
        "jackrabbits", "cardinal", "tommies", "aggies", "hurricane",
        "utes", "keydets", "huskies", "shockers", "terriers", "49ers",
        "roadrunners", "anteaters", "dragons", "camels", "grizzlies",
        "saints", "gamecocks", "hilltoppers", "blazers", "mocs", "pride",
        "great", "danes", "retrievers", "catamounts", "nittany",
        "wolverines", "hatters", "bulldog", "panther", "black",
        "bears", "catamount", "retriever", "wolverine",
        "bearcats", "terrapins", "buckeyes", "texans", "highlanders",
        "gauchos", "seahawks", "tribe", "lancers", "mustangs",
        "matadors", "braves", "griffins", "flyers", "pioneers", "colonels",
        "phoenix", "aces", "stags", "bengals", "vandals", "redbirds",
        "dolphins", "sharks", "jaspers", "foxes", "warriors", "bobcats",
        "racers", "chargers", "heels", "bison", "ospreys", "lumberjacks",
        "mavericks", "royals", "broncs", "peacocks", "salukis",
        "skyhawks", "seawolves", "volunteers", "trailblazers", "flames",
        "beacons", "tritons", "commodores", "rams", "hokies",
        "leathernecks", "dons", "hawks", "eagles", "screaming",
        "river", "riverhawks", "ramblers", "lakers", "flash",
        "purple", "gators", "tar",
        "razorbacks", "seminoles", "cyclones", "jayhawks", "hurricanes",
        "cornhuskers", "ducks", "boilermakers", "orange", "horned",
        "frogs", "cavaliers", "deacons", "demon", "badgers",
        "zips", "falcons", "chanticleers", "hens", "buccaneers",
        "antelopes", "rainbow", "flashes", "explorers", "minutemen",
        "raiders", "wolf", "spiders", "billikens", "bearkats", "coyotes",
        "hornets", "midshipmen", "warhawks", "bulls", "bisons",
        "chippewas", "ragin", "cajuns", "greyhounds", "hose",
        "thundering", "herd", "mastodons", "miners", "rattlers",
        "delta", "governors", "monarchs", "norse", "rockets",
        "penguins", "colonials", "privateers", "vaqueros",
        "islanders", "thunderbirds", "vikings", "mountain", "big",
    }

    parts = n.split()
    parts = [p for p in parts if p not in remove_words]

    # special cases / aliases
    aliases = {
        "ucla": "ucla",
        "rutgers": "rutgers",
        "st johns": "st johns",
        "nc state": "north carolina state",
        "smu": "smu",
        "usc": "usc",
        "southern california": "usc",
        "boise state": "boise state",
        "san diego state": "san diego state",
        "gw": "george washington",
        "george washington": "george washington",
        "south carolina upstate": "usc upstate",
        "state johns": "st johns",
        "state johns storm": "st johns",
        "cal baptist": "california baptist",
        "albany": "university at albany",
        "university at albany": "albany",
        "arkansaslittle rock": "little rock",
        "utarlington": "ut arlington",
        "n colorado": "northern colorado",
        "mt state marys": "mount st marys",
        "se missouri state": "southeast missouri state",
        "tennmartin": "tennessee martin",
        "tennmartin skyhawks": "tennessee martin",
        "tennesseemartin": "tennessee martin",
        "unc wilmington": "unc wilmington",
        "uic": "uic",
        "uc san diego": "uc san diego",
        "fairleigh dickinson": "fairleigh dickinson",
        "north carolina": "north carolina",
        "north carolina tar heels": "north carolina",
        "texas longhorns": "texas",
        "tennessee volunteers": "tennessee",
        "vcu rams": "vcu",
        "virginia tech hokies": "virginia tech",
        "loyola chi": "loyola chicago",
        "loyola chi ramblers": "loyola chicago",
        "state francis pa": "st francis pa",
        "state francis pa flash": "st francis pa",
        "evansville purple": "evansville",
        "fort wayne": "purdue fort wayne",
        "holy cross crusaders": "holy cross",
        "iupui": "iu indy",
        "lafayette leopards": "lafayette",
        "miss valley state": "mississippi valley state",
        "pennsylvania quakers": "pennsylvania",
        "prairie view": "prairie view am",
        "sam houston state": "sam houston",
        "se louisiana": "southeastern louisiana",
        "siuedwardsville": "siu edwardsville",
        "state bonaventure bonnies": "state bonaventure",
        "north carolina tar": "north carolina",
        "florida gators": "florida",
        "akron zips": "akron",
        "bowling falcons": "bowling",
        "coastal carolina chanticleers": "coastal carolina",
        "delaware hens": "delaware",
        "east tennessee state buccaneers": "east tennessee state",
        "florida intl": "florida international",
        "grand canyon antelopes": "grand canyon",
        "hawaii rainbow": "hawaii",
        "kent state flashes": "kent state",
        "la salle explorers": "la salle",
        "massachusetts minutemen": "massachusetts",
        "middle tennessee raiders": "middle tennessee",
        "nevada wolf": "nevada",
        "richmond spiders": "richmond",
        "saint louis billikens": "saint louis",
        "state louis": "saint louis",
        "sam houston state bearkats": "sam houston state",
        "san jose state": "san jose state",
        "san josé state": "san jose state",
        "san jos state": "san jose state",
        "miami": "miami oh",
        "southern": "southern utah",
        "se louisiana": "louisiana",
        "texas amcc": "texas am",
        "ul monroe": "le moyne",
        "south dakota coyotes": "south dakota",
        "arkansaspine bluff": "arkansas pine bluff",
        "marylandeastern shore": "maryland eastern shore",
        "mount state marys": "mount st marys",
    }

    key = " ".join(parts)
    key = aliases.get(key, key)

    # Handle CSU -> Cal State naming
    if key.startswith("csu "):
        key = "cal state " + key[4:]

    return key
