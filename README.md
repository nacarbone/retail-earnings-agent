# Building An Autonomous Earnings Trading Agent in the US Retail Industry</b>

## Contents

[1 ... Statement of Problem](#1-Statement-of-Problem)  
[2 ... Intended Client](#2-Intended-Client)  
[3 ... Key Business Question](#3-Key-Business-Question)  
[4 ... Data Sources](#4-Data-Sources)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Notes on Data Preparation](#Notes-on-Data-Preparation)  
[5 ... Business Impact of Work](#5-Business-impact-of-work)  
[6 ... How Model will be Used](#6-How-Model-will-be-Used)  
[7 ... Metric to Gauge Performance](#7-Metric-to-Gauge-Performance)  
[8 ... Methodology](#8-Methodology)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Basics](#Basics)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Environment Setup](#Environment-Setup)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Observation Space](#Observation-Space)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Action Space](#Action-Space)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Custom PPO Hyperparameters](#Custom-PPO-Hyperparameters)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Current Model Architecture](#Current-Model-Architecture)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Notes on Environment](#Environment)  
[9 ... Analysis of Results](#9-Analysis-of-Results)  
[10 ... Known Areas of Improvement](#10-Known-Areas-of-Improvment)  
[11 ... Further Research and Development](#11-Further-Research-and-Development)  
[Appendix](#12-Appendix)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Results of Code Testing Suite](#Results-of-Code-Testing-Suite)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Notes on Docker Container](#Notes-on-Docker-Container)

## 1 Statement of Problem

During earnings season, stocks tend to see periods containing some of their highest volume and volatility of the year. Analysts forecast companies' earnings and the market uses these to gauge whether a company performed better or worse than expected in a given quarter. Upon a report, investors take the opportunity re-assess a company's valuation. Many new investors may pour into a stock that reports favorable earnings, while others may leave for a stock that reports less-than-favorable earnings. 

Per the efficient market hypothesis, all known information should be priced into the stock prior to a company's earnings, so its difficult to predict whether or not a company will beat analyst estimates. However, there are predictable predictable phenomenon such as the volume and volatility surrounding these reports that may represent a profitable opportunity for a trading strategy that can identify these patterns.

The US retail industry was chosen for two reasons. First, there is seasonality in the earnings of retailers, which we hypothesize should lead to greater patterns in the earnings season. Secondly, we believe companies in this industry should be impacted by similar market forces. For instance, if consumer spending generally increases in a given year, we'd expect all of these companies to see a similar increase in their revenue.

## 2 Intended Client

An investment firm without a pre-existing algorithm to trade earnings reports, or one that would like to re-evaluate its current trading strategies for these periods.

## 3 Key Business Question

Are there patterns and information that could be used to predict trading activity in the periods surrounding earnings announcements for the US retail industry?

## 4 Data Sources

List of retail stock symbols: National Retail Federation (https://nrf.com/resources/top-retailers/top-100-retailers/top-100-retailers-2019)
Historical analyst estimates: WRDS IBES Database
Intraday Pricing Data: WRDS TAQ Database

#### Notes on Data Preparation

* The open, high, low, close, volume (OHCLV) data is normalized using the running means and standard deviations for the previous 10 days. This creates 
* The trading date is converted to an integer in [-1, 1] based on its relative days until the earnings are released.
* Estimate data is fed to the model raw.

## 5 Business Impact of Work

Consider the following 5 retailers listed in descending order of revenue for 2020:
* Walmart (WMT)
* Amazon (AMZN)
* Costco (COST)
* Walgreens (WBA)
* Kroger (KR)

Now suppose we have \$10,000 in our investment account and we would like to evenly apply this money across the companies to trade the Q3 2020 earnings season.

Within the 7 days following their Q3 2020 earnings reports, the companies saw the following percent deviations in their stock price, rounded to the nearest whole percent:

* Walmart: 4%  
* Amazon: 6%
* Costco: 2%
* Walgreens: -5% 
* Kroger: 3%

-- Source: Yahoo Finance

Suppose we had a perfect trading strategy, where we bought the stocks that increased and shorted those that decreased and took profit at these deviations. We could have earned a profit of:

(\$2,000 * 4\%) + ($2,000 * 6\%) + (\$2,000 * 2\%) + (\$2,000 * 5\%) + (\$2,000 * 3\%) = \$2,000

Or, more simply, a 20\% increase in the account value.

This is a best case scenario, whereas our more likely profit potential exists below this amount. 

## 6 How Model will be Used 

The model is designed to make its own decisions, so ideally it would be given access to its own investment account with permission to automate trades. The company could then feed a stream of data to the model through a trading day and the model. When in production, appropriate constraints should be in place; for instance, if the model were to lose more than 10\% of its portfolio value in a single day, it should be taken off line for researchers to study its performance and improve it for future usage.

## 7 Metric to Gauge Performance
Percent increase in account value in a given year

## 8 Methodology

### Basics

The study seeks to compare a reinforcement learning (RL) agent to a baseline model to gauge whether the RL agent achieves a substantive improvement over the baseline. The two models trade at one-minute intervals and make a decision on the at each and every one of those intervals.

At a glance, the baseline and RL models, respectively, are designed as follows: 
<ol>
    <li>At each timestep, the baseline model randomly selects an action to buy, sell or hold its current position. Based on this choice, it randomly selects an amount of shares to buy or sell based on its current cash balance and number of shares (i.e. the model can only sell as many shares as it actually has in its account. This process is similar to a random walk and would represent an investor whose trading decisions are completely arbitrary. This process is repeated over 100 episodes and the results averaged.</li>
    <li>At each timestep, the RL model is faced with the same set of decisions as the baseline model; however, it approaches its decisionmaking much more intelligently. Instead of guessing randomly, the model learns a policy by which it makes its actions. During the training process, the model probabilistically chooses to explore new actions or exploit its own knowledge of the underlying trading process. A neural network outputs an encoding of the current state to be fed into an action distribution. The action distribution consists of another neural network which produces the logits of its action type (buy/sell/hold) as well as the embedding of the amount to buy/sell/hold, conditioned on the action type. The embedding is then converted to logits using a fixed embedding matrix given by the environment. Any unavailble actions are masked out of the logits. The model selects actions from this distribution according to a policy, which is optimized using the Proximal Policy Optimization (PPO) algorithm.
</ol>

#### Environment Setup

* The agent trades 3-day periods beginning on the earnings date. So the agent will trade one day, the market will close, the company will report its earnings and the agent will trade the following two days after that.
* The agent is looking at earnings dates from 2010-2019. 2020 was excluded due to many abnormalities in company's earnings with the ongoing pandemic. The agent is trained on earnings dates from 2010-2017 and dates from 2018-2019 are reserved for testing.
* Only the top 5 retailers listed under the section "Business Impact of Work" are considered.

#### Observation Space
<ol>
    <li>A record of the values for open, high, low, close and volume for the stock's shares</li>
    <li>The offset
    <li>A record of the agent's previous actions</li>
    <li>A record of the agent's previous percentage of its cash invested</li>    
    <li>A record of the number of shares held in the agent's account</li>
    <li>Analyst estimates and their summary statistics for a particular earnings date</li>
    <li>The embedding matrix for the agent's actions</li>
    <li>The acceptable list of actions based on the accounts' position/li>
</ol>

#### Action Space
<ol>
    <li>Action type (e.g. buy, sell or hold)</li>
    <li>The amount of shares to buy or sell (a holding action type always corresponds to an amount of 0)</li>
</ol>

#### Custom PPO Hyperparameters
The model is built on top of Ray RLLib, which is an excellent library for distributed reinforcement learning. Using Ray exhaustive grid search was performed on a subset of the default PPO parameters and the following were found to perform best:

* clip_param (regulates the size of a policy update): 0.4
* kl_coeff (penalty for the KL divergence between the old and new policy): 0.04
* kl_target (target for the KL divergence): 0.01
* lambda (bias-variance tradeoff for using the current and future value estimates): 0.95
* gamma (discount factor applied to expected future rewards): 0.4
* lr (learning rate): 0.0005
* entropy_coeff (bonus given to encourage exploration): 0.03

#### Current Model Architecture

![alt text](assets/model_diagram.png) 

#### Notes on Environment

As with all RL algorithms, the environment is the main source of feedback for the model during training. Upon reset during training, the environment selects a new file from a list of input files, which should already be pre-processed and in json format. While the environment is useful for a training setting, its use of historical input files is not viable for a production setting where the data should be streamed. For this reason, the enviroment's functioning has been overridden in the application by the server.py file. The full training environment, however, has still been included in the repository as its required by Ray RLLib to restore an agent from training. This can hopefully be relaxed in the future when Ray implements methods to export its policies for PyTorch models.

## 9 Analysis of Results

At a high level, the model achieves similar performance to the baseline model, with both losing about 1\% on the testing data. We can examine some of the actions the models tend to take in the following graphs.

![alt text](assets/results_1.png) 
![alt text](assets/results_2.png) 

The first graph shows that the distribution of buying, selling and holding actions is roughly equal for the baseline model. This contrasts sharply with the RL model, which prefers to hold its current position most of the time. This mimics how real traders operate, as they typically wait until the right time to buy or sell before making ther decision. However, it could also be indicative of the model being too risk-averse and warrants more investigation. 

In the second graph, we can see that the RL model is much more judicious in its buying and sellings patterns than the baseline model, which prefers to buy and sell in larger quantities, thereby assuming more risk for changes to the market. 

We can now look at some graphs that detail the agent's performance in specific periods.

![alt text](assets/results_1.png) 

From the top chart, we see that the agent decides to build up its position heading into the earnings announcement, and then aggressively sells to take profit once the earnings are released. We see on the bottom chart that it earns the majority of the profit at this point in time, though its also able to capitalize on some upward movement later on in the period. As the agent heads into the earnings announcement, we see large orange blocks with little variation, showing the agent appears to have some certainty about what may happen. We think this demonstrates that the agent has learned an important feature about earnings announcements: that the stock price may very well spike when an announcement occurs.

![alt text](assets/results_2.png) 

Here we show the same chart, and again, the agent buys into the earnings announcement, but this time the price moves in the opposite direction and the agent take a large loss. However, whereas we saw large blocks leading into the announcement on the last chart, there is much more variation in the agent’s behavior here, suggesting a level of uncertainty about the announcement.

We should mention that all 

## 10 Known Areas of Improvement

## 11 Further Research and Development

## 12 Usage Instructions

The application is designed to run inside a Docker container. Please see the "Notes on building Docker container" section for information on the container's contents. From the top-level directory of the repository, the image can be built by running the following command in a shell:

```bash
sudo docker build . -t traderimg
```

Once the image has been built, you can then start the container with

```bash
sudo docker run -p 5000:5000 -it traderimg
```

You should then be given a shell inside the container and the application can be launched with the following command:
```bash
python app.py
```
The flask API will now be listening on port 5000.

As a side note, tests can be ran inside the container's shell with:
```bash
pytest tests/.
```

Please note that due to the computational complexity of the model, the time to complete these tests can vary greatly depending on the machine on which they're running.

From outside the container, please create a new environment with the library requests installed. This can as such:

```bash
conda create --name req-env python==3.8.5 requests
conda activate req-env
```

Now, assuming you're in the repository's top-level directory, from inside req-env you can run the following inside a python console or script to send the input-spec to the API:

```python
import json

import requests

with open('input-spec.json', 'r') as f:
    test_input = json.load(f)

headers = {'content-type' : 'application/json'}

r = requests.POST('http://0.0.0.0:5000/api/results', json=test_input, headers=headers)
```

Once the flask API returns a response, to get the output of the model, as well the agent's state, you can simply call ```r.json()```.

To get information on the history of the model's outputs based on the inputs its been sent, please visit http://0.0.0.0:5000/api/results in a web browser.

### Process Diagaram

![alt text](assets/architecture.png)

## Appendix

#### Results of Code Testing Suite

![alt text](assets/test-results.png)

![alt text](assets/test-coverage.png)

#### Notes on Docker Container

The Docker container performs the following tasks

<ol>
    <li>Installs system dependencies</li>
    <li>Downloads the model object from AWS</li>
    <li>Creates a conda environment from a .yml file committed to the repository</li>
    <li>Installs Python dependencies that wouldn’t be installable from requirements.txt to that conda environment</li>
    <li>Sets up the Docker container to run using the conda environment as its default shell</li>
</ol>

The idea behind Docker is that it provides a self-contained virtual operating system in which we can run an application. The rationale behind this is that different applications require different components to run, but with Docker we can know for certain that our application will run, provided the host computer has the ability to virtualize applications. In this sense, we can think of Docker as an abstract operating system.

For our purposes, we know that the application's installation would rely heavily on the use of conda environments, so I used a miniconda base image, which is specified at the top line of my Dockerfile.

```Docker
FROM continuumio/miniconda3
```

From there, most of the steps to build the container are accomplished through ```RUN``` commands. The ```RUN``` commands execute bash commands during the build process to add additional functionality to the container. Consider the following command in my Dockerfile:

```Docker
RUN apt-get -y install cmake
```

This command installs the cmake package to the container, which will be later used as dependency for the atari-py library.

Let's now look at the setup of the conda environment. We begin by copying the .yml file and creating an environment from this file:

```Docker
ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
```
Next, we want to setup the container so that its default shell uses our conda environment. This is done by appending the environment's path to our current path and adding an activate command to our ".bashrc" file. We see this here:

```Docker
RUN echo "source activate prod-env" > ~/.bashrc
ENV PATH /opt/conda/envs/prod-env/bin:$PATH
```

We must note that in the build stage, each ```RUN``` command represents a separate instance of the Docker shell. While our previous commands ensure the container will be started with our conda environment, they do not change the shell used in the build process. We want to install a few additional packages to our environment using ```RUN``` commands, so we can change the shell being used in the build process with the following
```Docker
SHELL ["conda", "run", "-n", "prod-env", "/bin/bash", "-c"]
```

Thereafter, we can install the other python packages we need like
```Docker
RUN yes | pip install 'ray[rllib]'
```
This will install ray into our conda environment.

Finally, because the application runs on port 5000, we need to tell Docker to expose this port for the container. This is done through the ```EXPOSE``` command and looks like this
```Docker
EXPOSE 5000
```

When built, our container will have everything it needs to run the application.