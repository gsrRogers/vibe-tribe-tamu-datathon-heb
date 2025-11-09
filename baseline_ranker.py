"""
BM25 Baseline Ranker for TAMU-25 Challenge
Simple keyword-based ranking using BM25 algorithm
"""
import json
from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Tuple

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def preprocess_text(text):
    """Simple text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep hyphens and spaces
    text = re.sub(r'[^a-z0-9\s-]', ' ', text)
    # Tokenize
    tokens = text.split()
    return tokens

def create_product_text(product):
    """Combine product fields for indexing"""
    parts = []

    # Title (most important, repeat 3x for boosting)
    if product.get('title'):
        parts.extend([product['title']] * 3)

    # Brand (repeat 2x)
    if product.get('brand'):
        parts.extend([product['brand']] * 2)

    # Category path
    if product.get('category_path'):
        # Split category path and add each level
        categories = product['category_path'].split('->')
        parts.extend([cat.strip() for cat in categories] * 2)

    # Description
    if product.get('description'):
        parts.append(product['description'])

    # Ingredients
    if product.get('ingredients'):
        parts.append(product['ingredients'])

    return ' '.join(parts)

class BM25Ranker:
    """BM25-based product ranker"""

    def __init__(self, products):
        """Initialize ranker with product catalog"""
        self.products = products
        self.product_ids = [p['product_id'] for p in products]

        # Create searchable text for each product
        print("Building search index...")
        product_texts = [create_product_text(p) for p in products]

        # Tokenize all products
        tokenized_products = [preprocess_text(text) for text in product_texts]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_products)
        print(f"Index built for {len(products)} products")

    def rank(self, query_text, top_k=50):
        """
        Rank products for a query

        Args:
            query_text: The search query
            top_k: Number of top results to return

        Returns:
            List of (product_id, score) tuples
        """
        # Tokenize query
        query_tokens = preprocess_text(query_text)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k product IDs and scores
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            product_id = self.product_ids[idx]
            score = scores[idx]
            results.append((product_id, score))

        return results

def generate_submission(ranker, queries, output_file, top_k=50):
    """
    Generate submission file

    Args:
        ranker: BM25Ranker instance
        queries: List of query dicts with 'query_id' and 'query'
        output_file: Path to save submission
        top_k: Number of products to rank per query (min 30)
    """
    submission = []

    print(f"\nGenerating rankings for {len(queries)} queries...")
    for i, query in enumerate(queries, 1):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(queries)} queries...")

        query_id = query['query_id']
        query_text = query['query']

        # Rank products
        rankings = ranker.rank(query_text, top_k=top_k)

        # Create submission entries
        for rank, (product_id, score) in enumerate(rankings, 1):
            submission.append({
                'query_id': query_id,
                'rank': rank,
                'product_id': product_id
            })

    print(f"Generated {len(submission)} ranked results")

    # Save submission
    save_json(submission, output_file)
    print(f"Submission saved to {output_file}")

    return submission

def main():
    """Main function"""
    print("=" * 80)
    print("BM25 BASELINE RANKER")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    products = load_json('data/products.json')
    queries_test = load_json('data/queries_synth_test.json')

    print(f"[OK] Loaded {len(products)} products")
    print(f"[OK] Loaded {len(queries_test)} test queries")

    # Initialize ranker
    ranker = BM25Ranker(products)

    # Generate submission for test queries
    submission = generate_submission(
        ranker,
        queries_test,
        'teams/team-gsr/submission.json',
        top_k=50  # Rank top 50 products per query (requirement is min 30)
    )

    print("\n" + "=" * 80)
    print("BASELINE COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Validate submission: py -m tamu25.cli.main validate ...")
    print("2. Evaluate on training set to estimate performance")
    print("3. Build improved ranker with embeddings or re-ranking")

if __name__ == '__main__':
    main()
