#!/bin/bash


function task() {
    # Suspend process of Vagrant
    cd $HOME/bdi/vagrant-k8s
    vagrant suspend
    echo "Suspend done"
    # run python script
    cd $HOME/hieppm/IrregularHeartbeatDetection/scripts
    conda activate py38
    python cnn_model.py &
}


while true; do
    HOUR=$(date +%H)
    echo "HOUR = $HOUR"
    if [ "$HOUR" == "18" ]; then
        task
        exit 0
    else
        echo "Wating until 18h00 ..."
        echo "Sleeping 60s"
        sleep 60
    fi
done