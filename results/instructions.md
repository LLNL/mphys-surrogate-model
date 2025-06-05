# Logging a new experiment
To add a new experiment/trained model analysis of interest, start by creating a new subdirectory with a descriptive title. Within the subdirectory, create a `README.md` that describes, at minimum:
- author / creator name
- commit hash and code snippets (if applicable) for reproducibility
- description of experiment purpose and results

The experiment directory can then contain figures and saved model states or other logged data. Please avoid saving large data files to github (e.g. common datasets, large log files); if necessary, store these instead on /p/vast1/ml-uphys and include the directory path in the readme.

## Note:
By default, `.pkl`, `.nc`, `.pth`, and `.npz` files are excluded from version control. When creating a new experiment, you will need to explicitly add your results if they use these extensions.