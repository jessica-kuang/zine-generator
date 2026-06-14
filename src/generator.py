import json
import anthropic
from pathlib import Path

# load all profile data
def load_profile_data():
    with open("data/profiles/cluster_labels.json") as f:
        clusters = json.load(f)
    with open("data/profiles/palette.json") as f:
        palette = json.load(f)
    with open("data/profiles/curated_reads.json") as f:
        reads = json.load(f)
    with open("data/profiles/brand_matches.json") as f:
        brands = json.load(f)
    return clusters, palette, reads, brands

# build context string
def build_context(clusters, palette):
    archetypes = []
    motifs = []
    moods = []

    for label in clusters["labels"].values():
        archetypes.append(label["archetype"])
        motifs.extend(label["motifs"])
        moods.append(label["mood"])

    context = f"""You are writing for a specific person based on careful observation of their taste data.
    Their aesthetic archetypes (clusters inferred from their actual images):
    {', '.join(archetypes)}

    Recurring motifs observed across their images:
    {', '.join(motifs[:8])}

    The mood of their images:
    {'; '.join(moods)}

    Their color palette: {palette['palette_name']}
    Palette mood: {palette['palette_mood']}
    Hex codes: {', '.join(palette['hex_codes'])}

    Write in second person. Be specific and observational — write as if you've been 
studying this person's images, not describing a generic aesthetic category. 
Tone: like a very chic, perceptive older sister. Essayistic, warm, occasionally 
poetic. Never generic. Never listicle energy. Concise — 2-4 sentences per node.
    """
    return context, archetypes, motifs

# generate archetype description
def generate_archetype(archetype, motifs, mood, context, client):
    message = client.messages.create(
        model = "claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""{context}

            Write a 2-3 sentence description of what the '{archetype}' cluster means 
specifically for this person, based on their motifs ({', '.join(motifs[:3])}) 
and mood ({mood}). Make it feel observed and specific, not categorical.

Respond with ONLY the description text, nothing else."""
        }]
    )
    return message.content[0].text.strip()

# generate brand description
def generate_brand_node(brand, context, client):
    message = client.messages.create(
        model = "claude-sonnet-4-20250514",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": f"""{context}
            Write 1-2 sentences explaining why {brand['name']} ({brand['description']}) 
belongs in this person's taste world. Connect it specifically to their 
archetypes and motifs. Feel personal, not like a brand description.

Respond with ONLY the text, nothing else."""
        }]
    )
    return message.content[0].text.strip()

# ---- generate read description ----
def generate_read_node(article, context, client):
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": f"""{context}

Write 1-2 sentences explaining why the article '{article['title']}' 
from {article['publication']} belongs in this person's taste world.
Connect it to their specific aesthetic and intellectual interests.
Feel like a personal recommendation from a friend, not a summary.

Respond with ONLY the text, nothing else."""
        }]
    )
    return message.content[0].text.strip()

# ---- generate feature essay ----
def generate_feature(archetypes, context, client):
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""{context}

Write a short feature essay (3-4 sentences) for this person's monthly issue.
It should feel like a letter written specifically to them — observational, 
warm, tied to their dominant aesthetic of '{archetypes[0]}' and '{archetypes[1]}'.
Something they'd read and feel deeply seen by.

Respond with ONLY the essay text, nothing else."""
        }]
    )
    return message.content[0].text.strip()

# ---- assemble graph content ----
def assemble_graph_content(clusters, palette, reads, brands, client):
    context, archetypes, motifs = build_context(clusters, palette)
    graph_content = {
        "archetypes": [],
        "brands": [],
        "reads": [],
        "feature": "",
        "palette": {
            "name": palette["palette_name"],
            "mood": palette["palette_mood"],
            "hex_codes": palette["hex_codes"]
        }
    }

    # generate archetype nodes
    print("generating archetype descriptions...")
    for label in clusters["labels"].values():
        print(f"  → {label['archetype']}")
        desc = generate_archetype(
            label["archetype"],
            label["motifs"],
            label["mood"],
            context,
            client
        )
        graph_content["archetypes"].append({
            "label": label["archetype"],
            "motifs": label["motifs"],
            "desc": desc
        })

    # generate brand nodes
    print("generating brand descriptions...")
    for category, cat_brands in brands.items():
        for brand in cat_brands:
            print(f"  → {brand['name']}")
            desc = generate_brand_node(brand, context, client)
            graph_content["brands"].append({
                "label": brand["name"],
                "category": category,
                "desc": desc,
                "url": brand.get("url", "")
            })

    # generate read nodes
    print("generating read descriptions...")
    for article in reads:
        print(f"  → {article['title']}")
        desc = generate_read_node(article, context, client)
        graph_content["reads"].append({
            "label": article["title"],
            "publication": article["publication"],
            "desc": desc,
            "url": article["link"],
            "score": article["score"]
        })

    # generate feature essay
    print("generating feature essay...")
    graph_content["feature"] = generate_feature(archetypes, context, client)
    print(f"\nfeature: {graph_content['feature'][:100]}...")

    return graph_content

# ---- save ----
def save_graph_content(content, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(content, f, indent=2)
    print(f"\nsaved graph content to {output_path}")

# ---- main ----
if __name__ == "__main__":
    client = anthropic.Anthropic()
    clusters, palette, reads, brands = load_profile_data()
    content = assemble_graph_content(clusters, palette, reads, brands, client)
    save_graph_content(content, "data/profiles/graph_content.json")

