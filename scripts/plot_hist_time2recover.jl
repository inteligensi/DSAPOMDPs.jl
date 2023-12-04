using Plots
using Plots.PlotMeasures
using DataFrames
using CSV

df = CSV.read("outputs/results_1000.csv", DataFrame)

policy_names = ["Random", "ExpertHOSP", "ExpertDSA", "DESPOT",]

#recovery rate for each policy
rows_needs_treatment = df[!, :need_treatment] .== true
rows_not_needs_treatment = df[!, :need_treatment] .== false

#plot the histogram of time2recover for each policy in a side by side plot
min_val = 0
max_val = 24
bins = range(min_val, max_val, length=25) 
y_min = 0
y_max = 1

x_ticks = range(min_val, max_val, length=5)
y_ticks = range(y_min, y_max, length=5)

# Create a plot with subplots for each policy
p = plot(
    layout = (2, 2), 
    size = (700, 400),
    right_margin = 8mm, 
    left_margin = 3mm,
    bottom_margin = 3mm)

for (i, pol_name) in enumerate(policy_names)
    histogram!(p[i], df[rows_needs_treatment, "time2recover_$pol_name"], bins=bins, normed=true, 
             minorgrid=true, xlim=(min_val, max_val), xticks=x_ticks, ylim=(y_min, y_max), title=pol_name, legend=false, 
             xlabel="Decision epoch", ylabel="Normalized density")
end

# Display the plot
plt = plot(p)
savefig(plt, "outputs/hist_time2recover.png")

