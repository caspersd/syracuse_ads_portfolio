

@app.command()
def train():
    """Run full BERT training pipeline."""
    trainer = BERTTrainer()
    trainer.run()


@app.command()
def evaluate():
    """Evaluate trained model."""
    trainer = BERTTrainer()
    trainer.load_datasets()
    trainer.build_model()
    trainer.evaluate_model()


if __name__ == "__main__":
    app()