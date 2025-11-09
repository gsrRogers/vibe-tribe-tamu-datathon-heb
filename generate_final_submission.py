"""
Generate final test submission using the best configuration
"""
import json
from semantic_ranker import HybridRanker, load_json

def main():
    print("=" * 80)
    print("GENERATING FINAL TEST SUBMISSION")
    print("=" * 80)

    # Load configuration
    print("\nLoading best configuration...")
    with open('final_config.json', 'r') as f:
        config = json.load(f)

    print(f"\nConfiguration:")
    print(f"  Ranker: {config['ranker_type']}")
    print(f"  Model: {config['model']}")
    print(f"  BM25 Weight: {config['bm25_weight']:.2f}")
    print(f"  Semantic Weight: {config['semantic_weight']:.2f}")
    print(f"  Training Score: {config['training_score']:.4f} ({config['training_score']*100:.2f}%)")
    print(f"  Improvement: {config['improvement_over_baseline']:+.2f}%")

    # Load data
    print("\nLoading data...")
    products = load_json('data/products.json')
    queries_test = load_json('data/queries_synth_test.json')

    print(f"[OK] Loaded {len(products)} products")
    print(f"[OK] Loaded {len(queries_test)} test queries")

    # Initialize ranker
    print(f"\nInitializing {config['ranker_type']} ranker...")
    ranker = HybridRanker(products, model_name=config['model'])

    # Generate predictions
    print(f"\nGenerating predictions for {len(queries_test)} test queries...")
    submission = []

    for i, query in enumerate(queries_test, 1):
        if i % 20 == 0:
            print(f"  Processed {i}/{len(queries_test)} queries...")

        query_id = query['query_id']
        query_text = query['query']

        # Rank products
        rankings = ranker.rank(
            query_text,
            top_k=50,
            bm25_weight=config['bm25_weight'],
            semantic_weight=config['semantic_weight']
        )

        # Create submission entries
        for rank, (product_id, score) in enumerate(rankings, 1):
            submission.append({
                'query_id': query_id,
                'rank': rank,
                'product_id': product_id
            })

    print(f"\n[OK] Generated {len(submission)} ranked results")

    # Save to team directory
    output_file = 'teams/VibeTribe/submission.json'
    with open(output_file, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"[OK] Submission saved to {output_file}")

    # Validate
    print("\nValidating submission...")
    import subprocess
    result = subprocess.run(
        ['py', '-m', 'tamu25.cli.main', 'validate',
         '--submission', output_file,
         '--team', 'VibeTribe'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("[OK] Validation PASSED")
        print("\n" + "=" * 80)
        print("FINAL SUBMISSION READY!")
        print("=" * 80)
        print(f"\nFile: {output_file}")
        print(f"Expected score: ~{config['training_score']*100:.2f}% (based on training evaluation)")
        print(f"\nYour optimized ranker is ready for leaderboard submission!")
    else:
        print("[ERROR] Validation FAILED")
        print(result.stdout)
        print(result.stderr)

if __name__ == '__main__':
    main()
