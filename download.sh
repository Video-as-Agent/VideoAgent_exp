if [ "$1" = "metaworld" ]; then
    wget -c https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/metaworld/model-305.pt
    mkdir -p results/mw
    mv model-305.pt results/mw/model-305.pt
    echo "Downloaded VideoAgent metaworld model"
elif [ "$1" = "online" ]; then
    wget -c https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/metaworld/model-3053083.pt
    mkdir -p results/mw
    mv model-3053083.pt results/mw/model-3053083.pt
    echo "Downloaded VideoAgent-online metaworld model"
elif [ "$1" = "suggestive" ]; then
    wget -c https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/metaworld/model-4307.pt
    mkdir -p results/mw
    mv model-4307.pt results/mw/model-4307.pt
    echo "Downloaded VideoAgent-suggestive metaworld model"
elif [ "$1" = "ithor" ]; then
    wget -c https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/ithor/thor-402.pt
    mkdir -p results/thor
    mv model-402.pt results/thor/model-402.pt
    echo "Downloaded VideoAgent ithor model"
elif [ "$1" = "bridge" ]; then
    wget -c https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/bridge/model-44.pt
    mkdir -p results/bridge
    mv model-44.pt results/bridge/model-44.pt
    echo "Downloaded VideoAgent bridge model"
else 
    echo "Options: {metaworld, online, suggestive, ithor, bridge}"
fi
