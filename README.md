# psykoda
psykoda [saikoːda]: Detect anomalous IP addresses from IDS log.

psykoda is an **alert screening tool for IDS users** (network security operators) based on machine learning.
IDSs usually generate so many alerts that security operators cannot manually investigate all of them, but the alerts might contain potential threats of cyber attacks.
This tool uses machine learning to analyze the alert log and detect most anomalous IP addresses.
This mitigates alert fatigue for security operators and extracts potential cyber attacks.

This software consists of a library and an application built on it.
See API Reference for library details.

## Application Usage
See [separate document for details](doc/source/app/index.md).

### Install
1. Install Python and Poetry.
2. Clone this repository.
3. Run `poetry install`.

### Launch
Run `poetry run psykoda -h` to see command line reference.

## Getting Started

### Install Dependencies

- [python](https://www.python.org/) 3.8.x
- [poetry](https://python-poetry.org/)

### Clone psykoda repository
```
git clone <FIXME: repository in github.com>
```

### Install psykoda
```
cd psykoda
poetry install
poetry run psykoda -h
```
In Windows, if you get `ImportError: Could not find the DLL(s) ‘msvcp140_1.dll’.`, download and install the [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 or 2019](https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0).

### Starting with example settings

```
poetry run psykoda --config example\config.json --date_from 2020-04-04 --date_to 2020-04-07
```
If successful, a graphical representation of the results is displayed.
- `X` represents anomaly IP address
- `▲` represents known false positive IP address

Detailed results are outputed to the directory specified in `config.json`'s `io.output.dir`.
- In this exmaple, `io.output.dir` is `./example/result/`

## Technology
* Semi-supervised anomaly detection: [Deep Semi-Supervised Anomaly Detection (Deep SAD)](https://openreview.net/forum?id=HkgH0TEYwH)
* Explanation of anomalies: [Shapley Additive Explanations (SHAP)](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)

## Contributing
Any type of contributions are welcome!
See [contributing guidelines](CONTRIBUTING.md) for details.

## License
[MIT: Copyright 2021 FUJITSU Limited](LICENSE)
