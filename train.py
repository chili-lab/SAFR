import argparse
from analyzer import SentimentAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Train Sentiment Analyzer")
    parser.add_argument("--dataset", type=str, choices=["imdb", "sst2"], default="sst2",
                        help="Select the dataset to use (imdb or sst2)")
    parser.add_argument("--importance_lambda", type=float, default=1,
                        help="Value for importance_lambda")
    parser.add_argument("--interaction_lambda", type=float, default=1,
                        help="Value for interaction_lambda")
    args = parser.parse_args()

    config = {
        "dataset": args.dataset,
        "importance_lambda": args.importance_lambda,
        "interaction_lambda": args.interaction_lambda
    }

    analyzer = SentimentAnalyzer(file_path="data", config=config)
    model = analyzer.run()
    print(model)

if __name__ == '__main__':
    main()
