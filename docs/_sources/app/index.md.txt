# Application Usage


## Installation
### Prerequisites
Install [Python](https://www.python.org/downloads/) and [Poetry](https://python-poetry.org/docs/).

### Installation Command
Clone this repository and run `poetry install` to install with dependencies.

## Usage
See `psykoda -h`.

This application is designed for periodical execution and manual inspection and response to anomalies.

```dot
digraph G
{
  {
    node [shape=box]
    log [label="IDS log"]
    el [label="Exclude Lists"]
    Config
    Anomalies
    FP [label="False Positives"]
    TP [label="True Positives"]
  }
  {
    node [shape=ellipse style=dashed]
    human [label="Manual Inspection"]
    response
  }
  {
    node [shape=ellipse]
    ees [label="extract-exclude-screening"]
    feature [label="Feature Extraction"]
    split [label="train-apply Split"]
    detection [label="training and detection"]
  }
  log -> ees
  el -> ees
  ees -> feature -> split -> detection -> Anomalies -> human -> FP -> feature
  human -> TP -> response
  Config:s -> {ees:e, feature:e, split:e, detection:e} [style=dotted]
}
```

A range of dates is required to execute detection.
For each date in the range, a model is trained with IDS log and False positives before that date.
Then, the model is applied with IDS log of that date to detect anomalies of that date.

### Input
#### IDS log
The main input of this application is IDS log files.
Each log record must include following fields:

* timestamp
* source IP address
* destination IP address
* destination port number
* IDS signature ID

type|support
:-- | :--
FS-CSV|fully supported
[snort-CSV](http://manual-snort-org.s3-website-us-east-1.amazonaws.com/node21.html#SECTION00366000000000000000)|partially tested for Snort 2.x
[snort-syslog](http://manual-snort-org.s3-website-us-east-1.amazonaws.com/node21.html#SECTION00361000000000000000)|on roadmap
Snort 3|on roadmap

#### Exclude lists
Pattern matching of field values can be used to exclude some part of log from whole analysis.

Data Type|Pattern
:--|:--
IP address|CIDR format
other|exact match

#### Labels
Known normal samples help semi-supervised anomaly detection to reduce false positives.
They can be provided as a list of (timestamp, source IP address)es.
Feature values for them will be constructed from corresponding log records saved in previous executions.

### Configuration
All configuration goes to a configuration file, whose path should be passed as `--path_config` required option.
Configuration file should be in JSON and include an object.

Refer to:
* API Reference for keys and definitions.
* `config.json` included in this repository for an example.
* [Best Practices for Working with Configuration in Python Applications](https://tech.preferred.jp/en/blog/working-with-configuration-in-python/)
  when modifying the example app to create your own app with configuration:
  Use [dataclasses](https://docs.python.org/3/library/dataclasses.html) to define configuration and [dacite](https://github.com/konradhalas/dacite) to convert from a `dict`.

### Output
Inside the output directory specified in config, a subdirectory will be created for each detection unit.

File|Description
:--|:--
`stats.json`|metadata
`report.csv`|anomalies detected
`plot_detection.png`|visualization

Log records for each anomaly will be also provided, for both manual inspection and use as known false positives in subsequent executions.
