## Week 1
- [x] Create a git repository (M5)
- [x] Make sure that all team members have write access to the GitHub repository (M5)
- [ ] Create a dedicated environment for your project to keep track of your packages (M2)
- [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
- [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
- [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
- [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies you are using (M2+M6)
- [ ] Remember to comply with good coding practices (pep8) while doing the project (M7)
- [x] Do a bit of code typing and remember to document essential parts of your code (M7)
- [ ] Setup version control for your data or part of your data (M8)
- [x] Add command-line interfaces and project commands to your code where it makes sense (M9)
- [x] Construct one or multiple Docker files for your code (M10)
- [ ] Build the Docker files locally and make sure they work as intended (M10)
- [x] Write one or multiple configuration files for your experiments (M11)
- [x] Use Hydra to load the configurations and manage your hyperparameters (M11)
- [ ] Use profiling to optimize your code (M12)
- [x] Use logging to log important events in your code (M14)
- [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
- [ ] Consider running a hyperparameter optimization sweep (M14)
- [ ] Use PyTorch Lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

## Week 2
- [x] Write unit tests related to the data part of your code (M16)
- [x] Write unit tests related to model construction and/or model training (M16)
- [x] Calculate the code coverage (M16)
- [ ] Get some continuous integration running on the GitHub repository (M17)
- [ ] Add caching and multi-OS/Python/PyTorch testing to your continuous integration (M17)
- [ ] Add a linting step to your continuous integration (M17)
- [x] Add pre-commit hooks to your version control setup (M18)
- [ ] Add a continuous workflow that triggers when data changes (M19)
- [ ] Add a continuous workflow that triggers when changes to the model registry are made (M19)
- [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
- [x] Create a trigger workflow for automatically building your Docker images (M21)
- [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
- [ ] Create a FastAPI application that can do inference using your model (M22)
- [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
- [ ] Write API tests for your application and set up continuous integration for these (M24)
- [ ] Load test your application (M24)
- [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
- [x] Create a frontend for your API (M26)

## Week 3
- [ ] Check how robust your model is towards data drifting (M27)
- [ ] Deploy to the cloud a drift detection API (M27)
- [ ] Instrument your API with a couple of system metrics (M28)
- [ ] Set up cloud monitoring of your instrumented application (M28)
- [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
- [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
- [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
- [ ] Play around with quantization, compilation, and pruning for your trained models to increase inference speed (M31)

## Extra
- [ ] Write some documentation for your application (M32)
- [ ] Publish the documentation to GitHub Pages (M32)
- [ ] Revisit your initial project description. Did the project turn out as you wanted?
- [ ] Create an architectural diagram over your MLOps pipeline
- [ ] Make sure all group members have an understanding of all parts of the project
- [ ] Upload all your code to GitHub
