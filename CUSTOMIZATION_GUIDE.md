# Customizable Evaluation Parameters Guide

## Overview
The AI Content Evaluator now supports customizable scoring parameters and weights, allowing you to tailor the evaluation to different hackathon themes and requirements.

## Theme-Based Configurations

### Available Themes
 
1. **Default (Balanced)**
   - Dimensions: Uniqueness (25%), Completeness (30%), Impact (30%), Ethics (5%)
   - Sections: Problem (30%), Solution (35%), Architecture (35%)

2. **Sustainability & Environment**
   - Dimensions: Uniqueness (20%), Completeness (25%), Impact (40%), Ethics (15%)
   - Sections: Problem (35%), Solution (30%), Architecture (35%)
   - Focus: Higher weight on environmental impact and ethical considerations

3. **Healthcare & Life Sciences**
   - Dimensions: Uniqueness (20%), Completeness (35%), Impact (30%), Ethics (15%)
   - Sections: Problem (30%), Solution (40%), Architecture (30%)
   - Focus: Emphasis on solution completeness and ethical considerations

4. **FinTech & Finance**
   - Dimensions: Uniqueness (30%), Completeness (35%), Impact (25%), Ethics (10%)
   - Sections: Problem (25%), Solution (35%), Architecture (40%)
   - Focus: Innovation and technical implementation

5. **Education & Learning**
   - Dimensions: Uniqueness (25%), Completeness (30%), Impact (35%), Ethics (10%)
   - Sections: Problem (30%), Solution (35%), Architecture (35%)
   - Focus: Educational impact and accessibility

6. **AI & Machine Learning**
   - Dimensions: Uniqueness (35%), Completeness (30%), Impact (25%), Ethics (10%)
   - Sections: Problem (25%), Solution (30%), Architecture (45%)
   - Focus: Technical innovation and architectural complexity

## Using the Web Interface

1. Upload your PPT files
2. Select the appropriate theme from the dropdown
3. Click "Start Evaluation"

The system will automatically apply the theme-specific weights and parameters.

## Command Line Usage

### Using Predefined Themes
```bash
python report_generator.py --input_dir ./data --theme sustainability
```

### List Available Themes
```bash
python report_generator.py --list_themes
```

### Custom Weights via Command Line
```bash
python report_generator.py --input_dir ./data \
  --uniqueness_weight 0.3 \
  --completeness_weight 0.4 \
  --impact_weight 0.2 \
  --ethics_weight 0.1 \
  --problem_weight 0.25 \
  --solution_weight 0.35 \
  --architecture_weight 0.4
```

### Using Custom Configuration File
```bash
python report_generator.py --input_dir ./data --config_file ./config_samples/custom_config.json
```

## Creating Custom Configuration Files

Create a JSON file with your custom weights:

```json
{
  "dimensions": {
    "uniqueness": 0.25,
    "Completeness of the solution": 0.30,
    "impact on the theme chosen": 0.35,
    "ethical consideration": 0.10
  },
  "section_weights": {
    "problem_statement": 0.30,
    "proposed_solution": 0.35,
    "technical_architecture": 0.35
  }
}
```

### Important Notes:
- All dimension weights should sum to 1.0
- All section weights should sum to 1.0
- The system will warn you if weights don't sum to 1.0
- Configuration files are saved alongside results for reproducibility

## Output Files

When evaluation completes, you'll get:
- `evaluation.csv` - Main results
- `evaluation_per_model.csv` - Individual model scores
- `evaluation_config.json` - Configuration used for this evaluation

## Scoring Dimensions

1. **Uniqueness**: Innovation and originality of the approach
2. **Completeness of the solution**: How complete and well-developed the solution is
3. **Impact on the theme chosen**: Relevance and potential impact on the specific theme
4. **Ethical consideration**: Consideration of ethical implications and responsible AI practices

## Section Evaluation

1. **Problem Statement**: Clear identification and articulation of the problem
2. **Proposed Solution**: Quality and feasibility of the proposed solution
3. **Technical Architecture**: Technical design, implementation details, and system architecture

## Best Practices

1. **Choose the right theme**: Select the theme that best matches your hackathon track
2. **Consistent evaluation**: Use the same theme for all submissions in a competition
3. **Document configuration**: Save and share the configuration file for transparency
4. **Validate weights**: Ensure all weights sum to 1.0 for fair comparison
5. **Test configurations**: Try different themes to see how they affect rankings

## Examples

### Sustainability Hackathon
Use the "sustainability" theme which emphasizes:
- Environmental impact (40% weight)
- Ethical considerations (15% weight)
- Problem identification (35% of overall score)

### Technical Innovation Competition
Use the "ai_ml" theme which emphasizes:
- Uniqueness and innovation (35% weight)
- Technical architecture (45% of overall score)
- Cutting-edge implementation

### Healthcare Challenge
Use the "healthcare" theme which emphasizes:
- Solution completeness (35% weight)
- Ethical considerations (15% weight)
- Robust solution development (40% of overall score)

This flexibility allows you to adapt the evaluation criteria to match the specific goals and values of your hackathon or competition.
