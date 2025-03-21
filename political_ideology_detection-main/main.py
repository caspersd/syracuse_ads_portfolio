import sys
import typer
from pathlib import Path

#import directories
from config import NEWSSD_DATA_DIR, OPINION_PROCESSED_DATA_DIR, MPQA_DIR, PLOT_DIR, OPINION_PROCESSED_DATA_DIR, BERT_MODEL_DIR, OPINION_RESULTS_DIR,OPINION_MODELS_DIR\
,IDEOLOGY_RAW_POLITICAL_STATEMENTS_DIR, IDEOLOGY_INTERIM_DATA_DIR, IDIOLOGY_PROCESSED_DIR, IDEOLOGY_FEATURES_DATA_DIR

# Import processing modules
from experiments.opinion_extraction.preprocess_data import preprocess_data #erroring out, need to fix
from experiments.opinion_extraction.Naive_Bayes_Experiments import run_naive_bayes_experiment

sys.path.append(str(Path(__file__).resolve().parent / "src"))
# Import Class Functions 
from src.opinion_extraction.data_plotter import DataPlotter
from src.opinion_extraction.BERTTrainer import BERTTrainer
from src.opinion_extraction.data_loader import DatasetLoader

print(sys.path)
from src.ideological_ranking.evaluate_ranking import eval_ranking

# Import Model
from models.political_analysis.rank_ideology import rank_politicians


# Initialize Typer app
app = typer.Typer()

@app.command(name="nb-experiment")

def nb_experiment():
    run_naive_bayes_experiment(
        processed_path=OPINION_PROCESSED_DATA_DIR,
        results_path=OPINION_RESULTS_DIR,
        model_save_path=OPINION_MODELS_DIR,
        mpqa_path=MPQA_DIR
    )


@app.command()
def preprocess_opinion_data():
    preprocess_data()


@app.command()
def plotfigures():
    plotter = DataPlotter(OPINION_PROCESSED_DATA_DIR,PLOT_DIR, MPQA_DIR)
    plotter.run_all_plots()


@app.command()
def train_bert_model():
    loader = DatasetLoader(OPINION_PROCESSED_DATA_DIR)
    _, bert_test, val_dataset = loader.load_bert_datasets()
    trainer = BERTTrainer(OPINION_PROCESSED_DATA_DIR, BERT_MODEL_DIR, OPINION_RESULTS_DIR)
    trainer.load_datasets()
    trainer.build_model()
    trainer.train_model(k=10)
    trainer.save_model()
    trainer.evaluate_model(bert_test)

############ Political Opinions ########

from src.ideological_ranking.preprocess_raw_opinions import extract_opinions

#Extract opinion dataset from raw text
#Note, this should be run on a GPU as predictions are too slow to be feasible on cpu
@app.command()
def extract_dataset():

    trainer = BERTTrainer(OPINION_PROCESSED_DATA_DIR, BERT_MODEL_DIR, OPINION_RESULTS_DIR)
    trainer.build_model()
    extract_opinions(input_directory=IDEOLOGY_RAW_POLITICAL_STATEMENTS_DIR, output_directory=IDEOLOGY_INTERIM_DATA_DIR, predict_function=trainer.predict)


from src.ideological_ranking.extract_features import filter_embed_senator_sentences
@app.command()
def extract_features():
    path_2022 = IDEOLOGY_INTERIM_DATA_DIR / "extracted_opinion_sentences_2022.csv"
    path_2023 = IDEOLOGY_INTERIM_DATA_DIR / "extracted_opinion_sentences_2023.csv"
    filter_embed_senator_sentences(path_2022, path_2023, IDIOLOGY_PROCESSED_DIR, IDEOLOGY_FEATURES_DATA_DIR)

from models.political_analysis.rank_ideology import rank_politicians
@app.command()
def rank_opinions():
    rank_politicians()

@app.command()
def eval_ideology_rank():
    eval_ranking(rank_politicians())



if __name__ == "__main__":
    app()
