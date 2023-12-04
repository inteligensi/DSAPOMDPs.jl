using Plots
using Plots.PlotMeasures
using DataFrames
using CSV

df = CSV.read("outputs/results_1000.csv", DataFrame)

policy_names = ["Random", "ExpertHOSP", "ExpertDSA", "DESPOT",]

#recovery rate for each policy
rows_needs_treatment = df[!, :need_treatment] .== true
rows_not_needs_treatment = df[!, :need_treatment] .== false
pol_name = "DESPOT"

recovery_rates = [(pol_name, mean(df[rows_needs_treatment, "treated_$pol_name"])) for pol_name in policy_names]
recovery_std = [(pol_name, std(df[rows_needs_treatment, "treated_$pol_name"])) for pol_name in policy_names]

unnecessary_treatment_rates = [(pol_name, mean(df[rows_not_needs_treatment, "treated_$pol_name"])) for pol_name in policy_names]
unnecessary_treatment_std = [(pol_name, std(df[rows_not_needs_treatment, "treated_$pol_name"])) for pol_name in policy_names]

disc_reward_mean = [(pol_name, mean(df[!, "rdisc_$pol_name"])) for pol_name in policy_names]
disc_reward_std = [(pol_name, std(df[!, "rdisc_$pol_name"])) for pol_name in policy_names]

disc_reward_mean_need_treatment = [(pol_name, mean(df[rows_needs_treatment, "rdisc_$pol_name"])) for pol_name in policy_names]
disc_reward_std_need_treatment = [(pol_name, std(df[rows_needs_treatment, "rdisc_$pol_name"])) for pol_name in policy_names]

disc_reward_mean_not_need_treatment = [(pol_name, mean(df[rows_not_needs_treatment, "rdisc_$pol_name"])) for pol_name in policy_names]
disc_reward_std_not_need_treatment = [(pol_name, std(df[rows_not_needs_treatment, "rdisc_$pol_name"])) for pol_name in policy_names]

time2recover_mean = [(pol_name, mean(df[rows_needs_treatment, "time2recover_$pol_name"])) for pol_name in policy_names]
time2recover_std = [(pol_name, std(df[rows_needs_treatment, "time2recover_$pol_name"])) for pol_name in policy_names]

#create a table to summarize the results