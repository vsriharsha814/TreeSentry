# Deforestation Detection Visualization Guide

This guide explains how to use the visualization components of the deforestation detection project to generate informative visualizations of your results.

## Overview of Visualization Capabilities

The project includes three main types of visualizations:

1. **Single-year predictions**: Shows deforestation detection results for individual years
2. **Year-to-year changes**: Visualizes changes in deforestation between consecutive years
3. **Multi-year progression**: Displays the overall trend of deforestation across all available years

## 1. Generating Prediction Visualizations

The prediction script (`src/predict_script.py`) generates visualizations for each year's deforestation predictions.

### Key Visualization Functions

- `create_visualization()`: Creates two visualizations for each year:
  - A composite RGB image with deforested areas highlighted in red
  - A heatmap showing deforestation probability

### Usage Steps

1. **Run the prediction script for all available years**:

```bash
python src/predict_script.py --config config.yaml \
                            --model_path models/unet_best.pth \
                            --input_dir data/raw \
                            --output_dir results/predictions
```

2. **View the generated visualizations**:
   - Navigate to `results/predictions/visualizations/`
   - Each year will have its own set of visualizations with the year in the filename:
     - `deforestation_visualization_2018.png` 
     - `deforestation_probability_2018.png`
     - etc.

## 2. Visualizing Deforestation Changes

The change visualization script (`src/visualize_changes.py`) analyzes changes between years and creates visualization of these changes.

### Key Visualization Functions

- `create_change_visualizations()`: Creates several visualizations:
  - Year-to-year change maps showing new deforestation, reforestation, and unchanged areas
  - Multi-year progression showing all years side by side
  - Deforestation timeline plot showing the trend over time

### Usage Steps

1. **Run the visualization script after generating predictions**:

```bash
python src/visualize_changes.py --input_dir results/predictions \
                               --output_dir results/change_analysis
```

2. **View the generated change visualizations**:
   - Navigate to `results/change_analysis/`
   - You'll find several types of visualizations:
     - `change_2018_to_2019.png`: Year-to-year change maps
     - `change_overall_2018_to_2021.png`: Overall change from first to last year
     - `multi_year_progression_2018_to_2021.png`: Side-by-side comparison of all years
     - `deforestation_timeline_2018_to_2021.png`: Trend plot over time

3. **Review change statistics**:
   - Open `change_statistics_2018_to_2021.txt` to see detailed statistics about the changes

## 3. Generating Comprehensive Reports

The report generation script (`src/generate_report.py`) compiles all visualizations into an HTML report.

### Usage Steps

1. **Run the report generation script**:

```bash
python src/generate_report.py --config config.yaml \
                             --predictions_dir results/predictions \
                             --evaluation_dir results/evaluation \
                             --changes_dir results/change_analysis \
                             --output_dir results/report
```

2. **View the generated report**:
   - Open `results/report/deforestation_report.html` in a web browser
   - The report contains all visualizations organized into sections with explanations

## Tips for Effective Visualization

1. **Ensure unique filenames**:
   - Always include the year in visualization filenames to avoid overwriting
   - For multi-year visualizations, include the year range in the filename

2. **Use consistent colormaps**:
   - Deforestation probability maps use `RdYlGn_r` (red = high probability, green = low)
   - Change maps use a custom colormap (red = new deforestation, green = reforestation, beige = unchanged)

3. **Include appropriate legends and titles**:
   - All visualizations should include a title indicating what year or years they represent
   - Change maps should include a legend explaining the color scheme
   - Add statistical information directly to the visualizations when appropriate

4. **Create comparison layouts**:
   - When comparing multiple years, use side-by-side layouts
   - For before/after comparisons, use consistent scales and orientations

5. **Highlight areas of interest**:
   - Consider adding annotations to highlight significant areas of deforestation
   - In the report, include zoomed-in views of critical areas

## Common Issues and Solutions

### Issue: Visualizations Being Overwritten

**Solution**:
- Ensure year is included in all filenames
- Use unique identifiers for each visualization
- Structure output directories to separate visualizations by year if needed

### Issue: Years Not Displaying Correctly in Visualizations

**Solution**:
- Verify that the year parameter is being passed correctly through all functions
- Check that filenames include the year information
- Ensure titles and labels correctly display the year information

### Issue: Inconsistent Color Scales

**Solution**:
- Use fixed value ranges for colormaps (e.g., `vmin=0, vmax=1` for probability maps)
- Apply the same colormap to all visualizations of the same type
- Include color scale legends on all visualizations

By following these guidelines, you'll produce clear, consistent, and informative visualizations that effectively communicate your deforestation detection results.
