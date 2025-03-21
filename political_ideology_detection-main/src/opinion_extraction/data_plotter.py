from pathlib import Path
import typer
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from loguru import logger
import spacy
from src.opinion_extraction.data_loader import DatasetLoader
from src.opinion_extraction.mpqa_processor import MPQAFeatureExtractor
import pandas as pd

app = typer.Typer()

class DataPlotter:
    """Handles visualization tasks for dataset analysis."""
    
    def __init__(self, input_path: Path, output_path: Path, mpqa_path: Path):
        self.output_path = output_path
        self.dataset_loader = DatasetLoader(input_path)
        self.opinions_train, self.opinions_test, self.opinions_val = self.dataset_loader.load_base_datasets()
        self.nlp = spacy.load('en_core_web_sm')
        mpqa = MPQAFeatureExtractor(mpqa_path)
        self.subjectivity_dict = mpqa.subjectivity_dict

    def plot_sentence_distribution(self):
        """Plots the distribution of objective and subjective sentences."""
        try:
            logger.info("Generating Sentence Distribution Plot...")
            train_counts = self.opinions_train['Label'].value_counts()
            test_counts = self.opinions_test['Label'].value_counts()
            val_counts = self.opinions_val['Label'].value_counts()

            labels = ['OBJ', 'SUBJ']
            x = np.arange(len(labels))
            width = 0.25

            fig, ax = plt.subplots()
            ax.bar(x - width, train_counts, width, label='Train')
            ax.bar(x, test_counts, width, label='Test')
            ax.bar(x + width, val_counts, width, label='Val')

            ax.set_ylabel('Count')
            ax.set_title('Distribution of Objective and Subjective Sentences')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            plt.savefig(self.output_path / "sentence_distribution.png")
            plt.close(fig)

            logger.success("Sentence Distribution Plot Generated Successfully.")
        except Exception as e:
            logger.error(f"Error generating Sentence Distribution Plot: {e}")

    def plot_wordclouds(self):
        """Generates and saves word clouds for subjective and objective sentences."""
        try:
            logger.info("Generating Wordclouds...")

            def generate_wordcloud(text):
                return WordCloud(width=800, height=800, background_color='white').generate(text)

            obj_train = ' '.join(self.opinions_train[self.opinions_train['Label'] == 0]['Sentence'])
            subj_train = ' '.join(self.opinions_train[self.opinions_train['Label'] == 1]['Sentence'])
            obj_test = ' '.join(self.opinions_test[self.opinions_test['Label'] == 0]['Sentence'])
            subj_test = ' '.join(self.opinions_test[self.opinions_test['Label'] == 1]['Sentence'])

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes[0, 0].imshow(generate_wordcloud(obj_train), interpolation='bilinear')
            axes[0, 0].set_title("Train Objective Sentences")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(generate_wordcloud(subj_train), interpolation='bilinear')
            axes[0, 1].set_title("Train Subjective Sentences")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(generate_wordcloud(obj_test), interpolation='bilinear')
            axes[1, 0].set_title("Test Objective Sentences")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(generate_wordcloud(subj_test), interpolation='bilinear')
            axes[1, 1].set_title("Test Subjective Sentences")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig(self.output_path / "wordclouds.png")
            plt.close(fig)

            logger.success("Wordclouds Generated Successfully.")
        except Exception as e:
            logger.error(f"Error generating wordclouds: {e}")
        
    def plot_pos_tag_distribution(self):
        """Plots the distribution of POS tags for objective and subjective sentences."""
        try:
            logger.info("Generating POS Tag Distribution...")

            def get_pos_tags(sentences):
                return sentences.apply(lambda s: [token.pos_ for token in self.nlp(s)])

            obj_train = get_pos_tags(self.opinions_train[self.opinions_train['Label'] == 0]['Sentence'])
            subj_train = get_pos_tags(self.opinions_train[self.opinions_train['Label'] == 1]['Sentence'])

            obj_counts = obj_train.apply(pd.Series).stack().value_counts()
            subj_counts = subj_train.apply(pd.Series).stack().value_counts()

            all_tags = set(obj_counts.index).union(set(subj_counts.index))
            obj_counts = obj_counts.reindex(all_tags, fill_value=0)
            subj_counts = subj_counts.reindex(all_tags, fill_value=0)

            positions = np.arange(len(all_tags))
            width = 0.4

            fig, ax = plt.subplots()
            ax.bar(positions - width/2, obj_counts, width, label="Objective", color='blue')
            ax.bar(positions + width/2, subj_counts, width, label="Subjective", color='orange')

            ax.set_xlabel("POS Tag")
            ax.set_ylabel("Frequency")
            ax.set_title("POS Tag Distribution")
            ax.set_xticks(positions)
            ax.set_xticklabels(all_tags, rotation=45, ha='right')
            ax.legend()

            plt.tight_layout()
            plt.savefig(self.output_path / "pos_distribution.png")
            plt.close(fig)

            logger.success("POS Tag Distribution Plot Generated Successfully.")
        except Exception as e:
            logger.error(f"Error generating POS tag distribution: {e}")

    def plot_sentence_length_boxplots(self):
        """Generates boxplots for sentence lengths in objective and subjective sentences."""
        try:
            logger.info("Generating Boxplots for Sentence Lengths...")

            def get_token_counts(sentences):
                return sentences.apply(lambda s: len(self.nlp(s)))

            obj_lengths = get_token_counts(self.opinions_train[self.opinions_train['Label'] == 0]['Sentence'])
            subj_lengths = get_token_counts(self.opinions_train[self.opinions_train['Label'] == 1]['Sentence'])

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.boxplot([obj_lengths, subj_lengths], vert=False, labels=['Objective', 'Subjective'])
            ax.set_xlabel("Sentence Length")
            ax.set_title("Boxplot of Sentence Lengths")

            plt.savefig(self.output_path / "sentence_length_boxplots.png")
            plt.close(fig)

            logger.success("Sentence Length Boxplots Generated Successfully.")
        except Exception as e:
            logger.error(f"Error generating Sentence Length Boxplots: {e}")

    def plot_mpqa_sentiment_histogram(self):
        """Generates a histogram for MPQA sentiment scores."""
        try:
            logger.info("Generating MPQA Sentiment Histogram...")
            def compute_sentiment_scores(sentences):
                scores = []
                for sentence in sentences:
                    tokens = [token.text for token in self.nlp(sentence)]
                    weak, strong = 0, 0
                    for token in tokens:
                        if token in self.subjectivity_dict:
                            strength = self.subjectivity_dict[token]
                            weak += 1 if strength == 'weaksubj' else 0
                            strong += 1 if strength == 'strongsubj' else 0
                    scores.append((weak + 2 * strong) / max(1, len(tokens)))  # Normalize
                return scores

            obj_scores = compute_sentiment_scores(self.opinions_train[self.opinions_train['Label'] == 0]['Sentence'])
            subj_scores = compute_sentiment_scores(self.opinions_train[self.opinions_train['Label'] == 1]['Sentence'])
            fig = plt.figure()
            common_xlim = [0,0.7]
            plt.figure(figsize=(10, 10))
            plt.subplot(2, 1, 1)
            plt.hist(obj_scores, bins=20, alpha=0.5, label="Objective", color='blue')
            plt.xlabel("Sentiment Score")
            plt.ylabel("Frequency")
            plt.xlim(common_xlim)
            plt.title('Distribution of MPQA Sentiment Scores for Objective Sentences')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.hist(subj_scores, bins=20, alpha=0.5, label="Subjective", color='orange')
            plt.xlabel("Sentiment Score")
            plt.ylabel("Frequency")
            plt.title('Distribution of MPQA Sentiment Scores for Subjective Sentences')
            plt.legend()


            plt.savefig(self.output_path / "mpqa_sentiment_histogram.png")
            plt.close(fig)

            logger.success("MPQA Sentiment Histogram Generated Successfully.")
        except Exception as e:
            logger.error(f"Error generating MPQA Sentiment Histogram: {e}")

    def run_all_plots(self):
        """Runs all available plots."""
        self.plot_sentence_distribution()
        self.plot_wordclouds()
        self.plot_pos_tag_distribution()
        self.plot_sentence_length_boxplots()
        self.plot_mpqa_sentiment_histogram()