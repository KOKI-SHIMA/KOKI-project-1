from pathlib import Path
import csv
import html


CSV_PATH = Path(
    "/workspace/anime-twin/output/clip_results.csv"
)

HTML_PATH = Path(
    "/workspace/anime-twin/output/result.html"
)


if not CSV_PATH.exists():
    raise RuntimeError(
        f"Result CSV not found: {CSV_PATH}"
    )


with CSV_PATH.open(
    "r",
    newline="",
    encoding="utf-8",
) as csv_file:
    results = list(
        csv.DictReader(csv_file)
    )


rows = []

for result in results:
    rank = int(result["rank"])

    character = result[
        "character"
    ].replace("_", " ").title()

    character = html.escape(character)

    similarity = float(
        result["cosine_similarity"]
    )

    rows.append(
        f"""
        <tr>
            <td>{rank}</td>
            <td>{character}</td>
            <td>{similarity:.4f}</td>
        </tr>
        """
    )


page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AnimeTwin Results</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
        }

        table {
            border-collapse: collapse;
            width: 600px;
        }

        th,
        td {
            border: 1px solid black;
            padding: 10px;
            text-align: left;
        }

        th {
            background-color: #eeeeee;
        }
    </style>
</head>

<body>
    <h1>AnimeTwin Results</h1>

    <table>
        <tr>
            <th>Rank</th>
            <th>Character</th>
            <th>Cosine Similarity</th>
        </tr>
""" + "\n".join(rows) + """
    </table>

    <p>
        Similarity is not a probability.
    </p>
</body>
</html>
"""


HTML_PATH.write_text(
    page,
    encoding="utf-8",
)

print(f"Result page saved: {HTML_PATH}")