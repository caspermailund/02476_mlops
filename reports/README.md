# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [ ] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [ ] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

2

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s170422, s174152, s194277, s233595, s244492

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used Albumentations in our project to perform image augmentations (see dataloader.py). We applied transformations like HorizontalFlip, RandomBrightnessContrast, Rotate, MotionBlur and GaussianBlur to introduce variability into the training data. These augmentations helped simulate different image conditions, such as lighting changes, rotations, and blurring, which improved the robustness of our CNN for image classification. By generating a larger and more diverse dataset, Albumentations allowed the CNN to generalize better, preventing overfitting and potentially improving its performance on unseen images.

One of the reasons we chose Albumentations is its ease of integration with PyTorch, as well as its optimized performance compared to other augmentation libraries. It provided a simple and efficient way to apply complex transformations with minimal computational overhead.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We managed dependencies for this project using a ```requirement.txt``` file, which keeps a list of all the necessary python packages and their respective versions required to reproduce the results. The use of this approach ensures that everyone contributing to the project can easily set up an identical environment, required for remote work. Further, it provides future developers with a guide on how to reproduce the results and creates an easy access to the dependencies.
In order to create the environment used in this project it is assumed that developers have python 3.12.7, git and conda installed locally on their machines.

To create the environment follow these instructions:
```bash
git clone https://github.com/caspermailund/02476_mlops/tree/main
cd 02476_mlops
conda create -n myenv python=3.12.7
conda activate myenv
pip install –r requirements.txt
pip install -e .
```

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

The project was initialized using the cookiecutter [template](https://github.com/SkafteNicki/mlops_template) provided in the course, and we aimed to maintain this structure throughout the development process.

Key modifications include, but are not limited to:
- Created `dataloader.py` in the `src` folder.
- Added `dvc` tools.
- Introduced a `log` folder.
- Included `dvc.yaml` to ensure result reproducibility.
- Placed CI/CD configuration files under `.github/workflows`.
- Added `cloudbuild.yaml` to the root directory.
- Filled out essentials scripts such as `data.py` and `model.py`.
- Added a notebook to demonstrate images and their labels.
- Listed dependencies required for generating the results in `requirements.txt`.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We aimed to follow good coding practices, keeping in mind Jeff Atwood’s quote: “Code tells you how; Comments tell you why,” which was highlighted in module M7. To support this, we added clear documentation at the beginning of key scripts, such as data.py and train.py, using triple-quoted docstrings (""") to describe the script’s purpose and functionality. Proper documentation and typing are crucial in larger projects as they enhance collaboration, reduce onboarding time for new developers, and minimize errors by making the code easier to understand and maintain.
The third party package "ruff" was used to ensure good formatting and used in the CI workflow to ensure that good formatting was maintained throughout the project.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total, 9 tests were implemented on 4 key functions of the pipeline; ```test_data.py```, ```test_dataloader.py```, ```test_model.py``` and ```test_training.py```.
```test_data.py``` tests that
-	the data contains the right amount of samples, in this case 14
-	the shape of the image ```(3, 256, 256)```
-	the labels are within the range ```[0, 13]```.
```test_dataloader``` tests that
-   all components of the dataloader are functioning
      - initalization, instantiation, getitem, transformation
```test_model.py``` tests that
-	the output images produced is of shape ```[1, 14]```.
```test_training.py``` tests that
-	output images are produced
-	the loss function produces non zero values
-	the optimizer is initialized correctly

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The coverage report is seen in table bellow:

======================= 9 passed, 40 warnings in 16.03s ========================
Name                                     Stmts   Miss  Cover   Missing
----------------------------------------------------------------------
src/image_classification/__init__.py         0      0   100%
src/image_classification/data.py            49      3    94%   33-34, 122
src/image_classification/dataloader.py      27      0   100%
src/image_classification/model.py           24      0   100%
src/image_classification/train.py           50      6    88%   101-108, 118
tests/__init__.py                            4      0   100%
tests/test_api.py                            0      0   100%
tests/test_data.py                          28      0   100%
tests/test_dataloader.py                    47      0   100%
tests/test_model.py                          7      0   100%
tests/test_training.py                      12      0   100%
----------------------------------------------------------------------
TOTAL                                      248      9    96%

It is seen that the total coverage in our testing is 96%.

The percentages are and indicator of how much of the coded lines is actually executed during the tests. However even with high percentages there might still be bugs present in parts of the code that are not completely tested or scenarios that haven’t been tested. The percentages are however a good indicator of how much code is covered in the tests and higher percentages are therefore preferred since it means more of the code has been tested which reduces the likelihood of bugs exciting in the code. E.g. line 118 might be the most crucial part of the training and is "untested" but one way to look at it, is that you only have to look hard at the code that is untestes in order to trust the code to be error free, and in that way higher coverage means less time looking for errors in the code.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Our workflow included the use of branches and pull requests to create separate workspaces for all developers. The use of branches ensured that implementing new features and fixing bugs was isolated, thereby preventing interference with the main coding base which reduces the risk of large merge conflicts that could be hard to solve. Using pull request in the workflow allowed us to merge changes form individual branches into the main branch, ensuring that the main branch was constantly up to date. This overall improved version control by allowing organized collaboration between developers while still letting multiple developers work on the same project simultaneously.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

Yes, we used DVC with a GCP bucket to manage data. It enabled version control, ensuring consistent datasets,
reproducibility, and seamless collaboration. The GCP bucket provided scalable storage, centralized access,
and the ability to revert to previous data versions, making our workflow more efficient and robust.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

Our continuous integration (CI) setup, powered by GitHub Actions, ensures the reliability and maintainability of our codebase. Key features include:

**Unit Testing**
Automated unit tests validate the functionality of key components like data loading, preprocessing, model training, and evaluation. Tests in the `tests/` directory are triggered on commits and pull requests.

**Linting**
We use `flake8` and `black` for linting and formatting, ensuring adherence to PEP 8 standards and early identification of potential issues.

**Multi-OS and Multi-Python Version Testing**
Our workflow tests compatibility across operating systems (Linux, macOS, Windows) and Python versions (3.8, 3.9, 3.10) to ensure portability.

**Caching**
To reduce runtime, we cache Python dependencies, saving the `pip` environment and minimizing redundant installations.

The CI pipeline is defined in `tests.yaml` under `.github/workflows/` and includes steps for dependency installation, testing (`pytest`), linting, and cross-platform compatibility checks.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured our experiments using a `config.yaml` file, which is loaded into `train.py` using Hydra. The configuration file allows us to easily modify for example `batch_size`, `epochs`, and `learning_rate` without changing `train.py` directly. This setup provides flexibility when running different experiments. However, the extent of experimentation is somewhat limited, as the `config.yaml` file was primarily created for demonstration purposes and therefore does not include many hyperparameters... To run an experiment, simply adjust the values in the config file and execute:

```bash
python train.py
```

In addition to the config file, we implemented a command-line interface (CLI) in `data.py`. The CLI allows us to easily specify the train/test split when running experiments by using the --test-size option. For example, to set a 70/30 train/test split, run:

```bash
python data.py --test-size 0.3
```

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

When running `data.py`, we log the used train/test split. In `train.py`, we log the model initialization, hyperparameters, and the average loss of each epoch. For demonstration purposes, we used two different logging modules in the two scripts: Python's built-in logging module and loguru's logging module. The log files are saved as `.txt` files in our `log` folder to ensure traceability.  Lastly, we save the trained model as an artifact in Weights & Biases, which would potentially allows us to track and compare different experiment runs over time.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

We log loss, epoch, and model as an artifact to track the training process. The loss helps us monitor performance and adjust training if needed, while the epoch shows the model's progress across iterations. Saving the model as an artifact allows us to version and retrieve it later for comparison or deployment. However, since the model has only completed one epoch, the data is not highly valuable at this point—it’s more of a demonstration of the process. These logs are key for tracking performance, optimizing hyperparameters, ensuring reproducibility, and enabling collaboration, making the training process transparent and easier to manage.

Loss and epochs:
![Loss and epochs](figures/w&b_1.png)

Model artifact:
![Model artifact](figures/w&b_2.png)

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

Buckets:
![Buckets](figures/Buckets.png)

The bucket contains:
![Buckets](figures/Buckets_pc_parts.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

GCP artifact registry:
![artifact_1](figures/gcp_artifact_registry_1.png)

![artifact_2](figures/gcp_artifact_registry_2.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

GCP cloud build history:
![GCP cloud build history](figures/build_history.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

Yes, we managed to train our model in the cloud using the Engine. We started by creating a virtual machine instance and connected our local computer. After installing the necessary dependencies on the virtual machine, we proceeded with training the model via the Engine. We chose the Engine because of its simplicity, which made it easy to set up and manage the training process.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

Yes, we managed to write an API for our model. We implemented an endpoint ```/predict/``` that allows users to upload their own image, which the model then classifies as a specific pc part. The endpoint also returns both the predicted class and the probabilities of each class. This was done by using FastAPI which is a python framework that is known for its high performance for building an API. We used a model that was trained locally and loaded it into an evaluation mode in order to produce consistent predictions. Even though the API worked the model itself struggled with producing good results and ended up predicting all images as the same class.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We managed to deploy the API locally. We used uvicorn to host the FastAPI application on a local machine which was run on a specified port ``` http://127.0.0.1:8000/```. This allowed user friendly platform for testing and interacting with the API endpoint ```/predict/```. By deploying locally we created an environment where we could debug and test the performance of the model and to see if it behaved consistently. However, even though the API worked more or less as intended we were unable to deploy the API in the cloud due to time constraints. If we had managed to deploy the API in the cloud we would have had a larger accessibility for remote work which is preferred when collaborating in a group.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We did not manage to perform unit testing or load testing on our API. However, if we had managed to do so we would start by ensuring that the API does in fact work as intended and handles request correctly. For testing the functionality we would need to create a test script that creates a client that can send requests to the API. In the test script we would need to include ```assert``` to verify the expected results. To test the API we would then have to run ```pytest test/script/location/test_script.py```.
For load test we would use the Locust framework. We would have to create a ``` locustfile.py``` file and then run the command ``` locust -f tests/script/location/locustfile.py``` and navigate to a host where we can simulate the amount of users for the load test.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

**Yes, we implemented a frontend for our API.**

We created a simple web interface that allows users to upload an image of a PC hardware component. Users can now:
1. **Upload an image**: The frontend provides an "Image Preview" feature so users can confirm the uploaded image.
2. **Interact with the interface locally**: The frontend can be run using the `npm start` command in the terminal, and it is accessible at `http://localhost:3000/`.

However, **the frontend does not yet classify the images.** While the upload and preview functionalities work, we need to integrate the backend classification logic with the frontend. This integration will ensure that uploaded images are sent to the backend for classification and the results are displayed to the user.

The reasons for building this frontend are:
- **Ease of use**: A web-based interface makes the tool accessible to non-technical users.
- **Testing and visualization**: It allows us to test the basic upload and preview features while we work on completing the backend integration.

Further work is required to complete the classification pipeline and fully connect the frontend with the backend API.

Frontend Interface:
![Frontend Interface for PC Hardware Image Classification](figures/image_upload_interface.png)


### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

Student s170422 ...

Student s174152 ...

Student s194277 ...

Student s233595 ...

Student s244492 ...

All members contributed to ...
Addititonally ChatGPT was used to help understanding errors and commands never used before.
