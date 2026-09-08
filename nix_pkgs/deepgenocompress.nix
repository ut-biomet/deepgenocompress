{
  buildPythonPackage,

  # python dependencies
  cyvcf2,
  numpy,
  pandas,
  scikit-learn,
  keras,
  keras_backend_pkg ? null,

  # tests
  pytestCheckHook,
  pytest-lazy-fixtures,
  pytest-mock,
  tensorflow,
  ...
}:
buildPythonPackage {
  pname = "deepgenocompress";
  version = "0.1.0";
  pyproject = true;
  doCheck = true;

  env = {
  };

  src = ./..;

  build-system = [
  ];

  dependencies = [
    cyvcf2
    numpy
    pandas
    scikit-learn
    keras
    keras_backend_pkg
  ];

  nativeCheckInputs = [
    pytestCheckHook
    pytest-lazy-fixtures
    pytest-mock
    tensorflow
  ];

  pythonImportsCheck = [
    "deepgenocompress"
    "deepgenocompress.utils"
    "deepgenocompress.exceptions"
    "deepgenocompress.warnings"
  ];

  disabledTests = [
  ];

  meta = {
    homepage = "https://github.com/ut-biomet/DeepCGP";
    description = "An AI-Driven Tool for Compressing Genome-Wide Polymorphisms in Plant Breeding";
  };
}
