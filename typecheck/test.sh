export PYTHON_VERSION=3.8

function pyre()
{
    cp pyproject.toml ./typecheck/pyre &&\
    docker build \
        -t pyrepsykoda \
        --build-arg HTTP_PROXY \
        --build-arg HTTPS_PROXY \
        --build-arg http_proxy \
        --build-arg https_proxy \
        --build-arg PYTHON_VERSION \
        typecheck/pyre &&\
    rm -rf /.pyre/*
    docker run \
        -v $(pwd):/psykoda/:ro \
        -v $(pwd)/.pyre/:/psykoda/.pyre \
        -w /psykoda/ \
        pyrepsykoda pyre --source-directory src --search-path /usr/local/lib/python${PYTHON_VERSION}/site-packages
}

function pyright()
{
    echo "pyright: not supported yet"
    exit 1
}

function pytype()
{
    echo "pytype: not supported yet"
    exit 1
}

function usage()
{
    cat <<EOF
commands:
    pyre
    pyright
    pytype
EOF
}

function main()
{
    case $1 in
    "pyre")
        pyre
        ;;
    "pyright")
        pyright
        ;;
    "pytype")
        pytype
        ;;
    *)
        usage
        ;;
    esac
}

main "$@"
