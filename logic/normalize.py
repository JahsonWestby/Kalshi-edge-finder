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
        "north carolina tar": "north carolina",
        "florida gators": "florida",
    }

    key = " ".join(parts)
    key = aliases.get(key, key)

    # Handle CSU -> Cal State naming
    if key.startswith("csu "):
        key = "cal state " + key[4:]

    return key
