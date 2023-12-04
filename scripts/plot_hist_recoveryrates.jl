using Plots
using Plots.PlotMeasures
using DataFrames
using CSV

df = CSV.read("outputs/results_1000.csv", DataFrame)

policy_names = ["Random", "ExpertHOSP", "ExpertDSA", "DESPOT",]

#recovery rate for each policy
rows_needs_treatment = df[!, :need_treatment] .== true
rows_not_needs_treatment = df[!, :need_treatment] .== false

disc_reward_mean = [(pol_name, mean(df[!, "rdisc_$pol_name"])) for pol_name in policy_names]
disc_reward_std = [(pol_name, std(df[!, "rdisc_$pol_name"])) for pol_name in policy_names]

disc_reward_mean_need_treatment = [(pol_name, mean(df[rows_needs_treatment, "rdisc_$pol_name"])) for pol_name in policy_names]
disc_reward_std_need_treatment = [(pol_name, std(df[rows_needs_treatment, "rdisc_$pol_name"])) for pol_name in policy_names]

disc_reward_mean_not_need_treatment = [(pol_name, mean(df[rows_not_needs_treatment, "rdisc_$pol_name"])) for pol_name in policy_names]
disc_reward_std_not_need_treatment = [(pol_name, std(df[rows_not_needs_treatment, "rdisc_$pol_name"])) for pol_name in policy_names]


#plot the histogram of the disc_reward for each policy in a side by side plot
min_val = -1e5 #minimum([minimum(df[!, "rdisc_$pol_name"]) for pol_name in policy_names])
max_val = 50000 #maximum([maximum(df[!, "rdisc_$pol_name"]) for pol_name in policy_names])
bins = range(min_val, max_val, length=30) 
y_min = 0
y_max = 0.00018

x_ticks = range(min_val, max_val, length=3)
y_ticks = range(y_min, y_max, length=3)

# Create a plot with subplots for each policy
p = plot(
    layout = (2, 2), 
    size = (700, 400),
    # bottom_margin = 5mm,
    # left_margin = 5mm,
    # top_margin = 5mm,
    right_margin = 8mm,
    sharey = true,
    sharex = true)

for (i, pol_name) in enumerate(policy_names)
    histogram!(p[i], df[!, "rdisc_$pol_name"], bins=bins, normed=true, 
             minorgrid=true, xlim=(min_val, max_val), xticks=x_ticks, ylim=(y_min, y_max), title=pol_name, legend=false)
end

# Display the plot
plt = plot(p)
savefig(plt, "outputs/hist_rdisc.pdf")

