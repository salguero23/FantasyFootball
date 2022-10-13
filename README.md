# NFL Fantasy Football Projections - Ensemble Methods to Machine & Reinforcement Learning

## Introduction
In fantasy football leagues managers are required to draft new players before the start of the season. With the chaotic nature that is the NFL it is difficult to have an understanding on which players will perform well and who will have mediocre/horrible seasons. For most fantasy managers, they rely on qualitative analysis and online projections generated from third-party cites on who will be the best players for the upcoming season. Although it is possible to find success using these methods, we must ask ourselves can we improve this workflow and with a certain degree of confidence can we make accurate projections on which players will outperform for the upcoming season?

As of October 13, 2022, this repository is still a work in progress. However, we can still discuss the work we have so far and future goals we have for the project. It should be noted at this time that the data gathered is strictly for Flex Players i.e., Running Backs, Wide Receivers, and Tight Ends.

Rather than using only one model, the goal is to have an ensemble of multiple machine and reinforcement learning models project next year’s fantasy points scored. Some models settled upon are a Ridge Regression, Random Forest, and Support Vector Machine. Although the reinforcement learning side of the project is still in early development, so far, I have settled upon using a Proximal Policy Optimization (PPO) algorithm as our learning agent.


## Table of Contents



Data was collected via https://www.pro-football-reference.com/

Work in progress dashboard(s)
- https://public.tableau.com/app/profile/andrew.salguero/viz/NFLData_16643117180950/Dashboard1
