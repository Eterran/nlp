"""
Evaluation pipeline for Multilingual News Article Summarizer.

Evaluates model performance on:
- CNN/DailyMail (English)
- MLSUM French
- MLSUM German

Uses 100 test samples per language and calculates ROUGE-1, ROUGE-2, and ROUGE-L scores.
"""

import csv
import os
from typing import List, Dict, Any
from datasets import load_dataset
from rouge_score import rouge_scorer
from summariser import Summarizer


class SummarizerEvaluator:
    """Evaluator for multilingual news article summarization."""

    def __init__(self):
        """Initialize the evaluator with required models."""
        print("Initializing Summarizer...")
        self.summarizer = Summarizer()
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        print("Evaluator initialized successfully.\n")

    def load_dataset_samples(
        self, dataset_name: str, language: str, num_samples: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Load samples from a dataset using streaming to avoid memory issues.

        Args:
            dataset_name: Name of the dataset to load
            language: Language identifier
            num_samples: Number of samples to load (default: 100)

        Returns:
            List of sample dictionaries with 'article' and 'summary' keys
        """
        print(f"Loading {num_samples} samples from {dataset_name} ({language})...")

        samples = []
        successful_loads = 0

        try:
            # Configure dataset loading
            if dataset_name == "cnn_dailymail":
                dataset = load_dataset(
                    dataset_name, "3.0.0", streaming=True, split="test"
                )
                article_key = "article"
                summary_key = "highlights"
            elif dataset_name == "mlsum":
                # MLSUM needs trust_remote_code=True
                dataset = load_dataset(
                    dataset_name,
                    language,
                    streaming=True,
                    split="test",
                    trust_remote_code=True,
                )
                article_key = "text"
                summary_key = "summary"
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # Stream and collect samples until we get exactly num_samples successful ones
            for sample in dataset:
                try:
                    # Extract article and summary
                    article = sample.get(article_key, "").strip()
                    summary = sample.get(summary_key, "").strip()

                    # Skip if either article or summary is empty
                    if not article or not summary:
                        continue

                    samples.append({"article": article, "summary": summary})

                    successful_loads += 1  # Progress logging every 10 samples
                    if successful_loads % 10 == 0:
                        print(f"  Loaded {successful_loads}/{num_samples} samples...")

                    # Stop when we have enough samples
                    if successful_loads >= num_samples:
                        break

                except Exception as e:
                    print(f"  Warning: Skipping sample due to error: {e}")
                    continue

            print(
                f"Successfully loaded {len(samples)} samples from {dataset_name} ({language})\n"
            )
            return samples

        except Exception as e:
            print(f"Error loading dataset {dataset_name} ({language}): {e}")
            return []

    def evaluate_sample(self, article: str, reference_summary: str) -> Dict[str, Any]:
        """
        Evaluate a single article-summary pair.

        Args:
            article: Input article text
            reference_summary: Ground truth summary

        Returns:
            Dictionary with evaluation results
        """
        try:
            # Generate summary using the summarizer
            result = self.summarizer.summarize(article)

            if result["error"]:
                return {
                    "success": False,
                    "error": result["error"],
                    "rouge1_f": 0.0,
                    "rouge2_f": 0.0,
                    "rougeL_f": 0.0,
                    "generated_summary": None,
                    "detected_language": result.get("detected_language_ld"),
                }

            generated_summary = result["final_summary"]
            if not generated_summary:
                return {
                    "success": False,
                    "error": "No summary generated",
                    "rouge1_f": 0.0,
                    "rouge2_f": 0.0,
                    "rougeL_f": 0.0,
                    "generated_summary": None,
                    "detected_language": result.get("detected_language_ld"),
                }

            # Calculate ROUGE scores
            scores = self.rouge_scorer.score(reference_summary, generated_summary)

            return {
                "success": True,
                "error": None,
                "rouge1_f": scores["rouge1"].fmeasure,
                "rouge2_f": scores["rouge2"].fmeasure,
                "rougeL_f": scores["rougeL"].fmeasure,
                "generated_summary": generated_summary,
                "detected_language": result.get("detected_language_ld"),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "rouge1_f": 0.0,
                "rouge2_f": 0.0,
                "rougeL_f": 0.0,
                "generated_summary": None,
                "detected_language": None,
            }

    def evaluate_dataset(
        self, dataset_name: str, language: str, num_samples: int = 25
    ) -> Dict[str, Any]:
        """
        Evaluate summarizer on a complete dataset.

        Args:
            dataset_name: Name of the dataset
            language: Language identifier
            num_samples: Number of samples to evaluate

        Returns:
            Dictionary with aggregated results
        """
        print(f"=== Evaluating {dataset_name} ({language}) ===")

        # Load samples
        samples = self.load_dataset_samples(dataset_name, language, num_samples)
        if not samples:
            return {
                "dataset": dataset_name,
                "language": language,
                "total_samples": 0,
                "successful_evaluations": 0,
                "avg_rouge1_f": 0.0,
                "avg_rouge2_f": 0.0,
                "avg_rougeL_f": 0.0,
                "individual_results": [],
                "error": "Failed to load samples",
            }

        # Evaluate each sample
        individual_results = []
        successful_evaluations = 0
        total_rouge1 = 0.0
        total_rouge2 = 0.0
        total_rougeL = 0.0

        for i, sample in enumerate(samples):
            try:  # Progress logging every 10 evaluations
                if (i + 1) % 10 == 0:
                    print(f"  Evaluating sample {i + 1}/{len(samples)}...")

                # Evaluate single sample
                eval_result = self.evaluate_sample(sample["article"], sample["summary"])

                # Store individual result
                individual_result = {
                    "sample_id": i + 1,
                    "dataset": dataset_name,
                    "language": language,
                    "success": eval_result["success"],
                    "error": eval_result["error"],
                    "rouge1_f": eval_result["rouge1_f"],
                    "rouge2_f": eval_result["rouge2_f"],
                    "rougeL_f": eval_result["rougeL_f"],
                    "detected_language": eval_result["detected_language"],
                    "reference_summary": (
                        sample["summary"][:200] + "..."
                        if len(sample["summary"]) > 200
                        else sample["summary"]
                    ),
                    "generated_summary": (
                        eval_result["generated_summary"][:200] + "..."
                        if eval_result["generated_summary"]
                        and len(eval_result["generated_summary"]) > 200
                        else eval_result["generated_summary"]
                    ),
                }
                individual_results.append(individual_result)

                # Accumulate scores for successful evaluations
                if eval_result["success"]:
                    successful_evaluations += 1
                    total_rouge1 += eval_result["rouge1_f"]
                    total_rouge2 += eval_result["rouge2_f"]
                    total_rougeL += eval_result["rougeL_f"]

            except Exception as e:
                print(f"  Warning: Error evaluating sample {i + 1}: {e}")
                individual_results.append(
                    {
                        "sample_id": i + 1,
                        "dataset": dataset_name,
                        "language": language,
                        "success": False,
                        "error": str(e),
                        "rouge1_f": 0.0,
                        "rouge2_f": 0.0,
                        "rougeL_f": 0.0,
                        "detected_language": None,
                        "reference_summary": (
                            sample["summary"][:200] + "..."
                            if len(sample["summary"]) > 200
                            else sample["summary"]
                        ),
                        "generated_summary": None,
                    }
                )
                continue

        # Calculate averages
        avg_rouge1 = (
            total_rouge1 / successful_evaluations if successful_evaluations > 0 else 0.0
        )
        avg_rouge2 = (
            total_rouge2 / successful_evaluations if successful_evaluations > 0 else 0.0
        )
        avg_rougeL = (
            total_rougeL / successful_evaluations if successful_evaluations > 0 else 0.0
        )

        print(
            f"Completed evaluation: {successful_evaluations}/{len(samples)} successful"
        )
        print(f"Average ROUGE-1: {avg_rouge1:.4f}")
        print(f"Average ROUGE-2: {avg_rouge2:.4f}")
        print(f"Average ROUGE-L: {avg_rougeL:.4f}\n")

        return {
            "dataset": dataset_name,
            "language": language,
            "total_samples": len(samples),
            "successful_evaluations": successful_evaluations,
            "avg_rouge1_f": avg_rouge1,
            "avg_rouge2_f": avg_rouge2,
            "avg_rougeL_f": avg_rougeL,
            "individual_results": individual_results,
            "error": None,
        }

    def run_full_evaluation(self, num_samples: int = 25) -> str:
        """
        Run complete evaluation pipeline on all datasets.

        Args:
            num_samples: Number of samples per dataset

        Returns:
            Path to the saved results CSV file
        """
        print(
            "🚀 Starting Full Multilingual Evaluation Pipeline 🚀\n"
        )  # Define datasets to evaluate
        datasets_config = [
            {"dataset_name": "cnn_dailymail", "language": "en"},
            {"dataset_name": "mlsum", "language": "fr"},
        ]

        all_summary_results = []

        # Evaluate each dataset
        for config in datasets_config:
            dataset_result = self.evaluate_dataset(
                config["dataset_name"], config["language"], num_samples
            )

            # Save individual CSV immediately after each language evaluation
            individual_csv_path = self.save_individual_results_to_csv(
                dataset_result["individual_results"],
                dataset_result["dataset"],
                dataset_result["language"],
            )
            print(
                f"✅ Saved {dataset_result['language'].upper()} results to: {individual_csv_path}"
            )

            # Store summary results
            all_summary_results.append(
                {
                    "dataset": dataset_result["dataset"],
                    "language": dataset_result["language"],
                    "total_samples": dataset_result["total_samples"],
                    "successful_evaluations": dataset_result["successful_evaluations"],
                    "success_rate": (
                        dataset_result["successful_evaluations"]
                        / dataset_result["total_samples"]
                        if dataset_result["total_samples"] > 0
                        else 0.0
                    ),
                    "avg_rouge1_f": dataset_result["avg_rouge1_f"],
                    "avg_rouge2_f": dataset_result["avg_rouge2_f"],
                    "avg_rougeL_f": dataset_result["avg_rougeL_f"],
                    "error": dataset_result["error"],
                }
            )

        # Save combined summary CSV
        summary_csv_path = self.save_summary_results_to_csv(
            all_summary_results
        )  # Print final summary
        print("📊 FINAL EVALUATION SUMMARY 📊")
        print("=" * 50)
        for result in all_summary_results:
            print(f"{result['dataset']} ({result['language'].upper()}):")
            print(
                f"  Success Rate: {result['success_rate']:.1%} ({result['successful_evaluations']}/{result['total_samples']})"
            )
            print(f"  ROUGE-1: {result['avg_rouge1_f']:.4f}")
            print(f"  ROUGE-2: {result['avg_rouge2_f']:.4f}")
            print(f"  ROUGE-L: {result['avg_rougeL_f']:.4f}")
            print()

        print(f"✅ Combined summary saved to: {summary_csv_path}")
        return summary_csv_path

    def save_individual_results_to_csv(
        self, individual_results: List[Dict], dataset_name: str, language: str
    ) -> str:
        """
        Save individual evaluation results to a language-specific CSV file with summary row.

        Args:
            individual_results: List of individual sample results
            dataset_name: Name of the dataset
            language: Language identifier

        Returns:
            Path to the saved CSV file
        """
        # Create filename
        csv_path = f"evaluation_results_{language}.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            if individual_results:
                fieldnames = individual_results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(individual_results)

                # Add summary row
                successful_results = [r for r in individual_results if r["success"]]
                if successful_results:
                    avg_rouge1 = sum(r["rouge1_f"] for r in successful_results) / len(
                        successful_results
                    )
                    avg_rouge2 = sum(r["rouge2_f"] for r in successful_results) / len(
                        successful_results
                    )
                    avg_rougeL = sum(r["rougeL_f"] for r in successful_results) / len(
                        successful_results
                    )

                    summary_row = {
                        "sample_id": "SUMMARY",
                        "dataset": dataset_name,
                        "language": language,
                        "success": f"{len(successful_results)}/{len(individual_results)}",
                        "error": None,
                        "rouge1_f": avg_rouge1,
                        "rouge2_f": avg_rouge2,
                        "rougeL_f": avg_rougeL,
                        "detected_language": None,
                        "reference_summary": "AVERAGE SCORES",
                        "generated_summary": f"Success Rate: {len(successful_results)/len(individual_results):.1%}",
                    }
                    writer.writerow(summary_row)

        return csv_path

    def save_summary_results_to_csv(self, summary_results: List[Dict]) -> str:
        """
        Save aggregated summary results to CSV file.

        Args:
            summary_results: List of aggregated dataset results

        Returns:
            Path to the saved CSV file
        """
        summary_csv_path = "evaluation_results_summary.csv"
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            if summary_results:
                fieldnames = summary_results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_results)

        return summary_csv_path


def main():
    """Main evaluation function."""
    try:
        evaluator = SummarizerEvaluator()
        csv_path = evaluator.run_full_evaluation(num_samples=25)
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"Results available in: {csv_path}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
