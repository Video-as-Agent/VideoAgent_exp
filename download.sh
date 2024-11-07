if [ "$1" = "metaworld" ]; then
    wget -c https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/metaworld/model-305.pt
    mkdir -p ckpts/metaworld
    mv model-305.pt ckpts/metaworld/model-305.pt
    echo "Downloaded VideoAgent metaworld model"
elif [ "$1" = "online" ]; then
    wget -c https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/metaworld/model-3053083.pt
    mkdir -p ckpts/metaworld
    mv model-3053083.pt ckpts/metaworld/model-3053083.pt
    echo "Downloaded VideoAgent-online metaworld model"
elif [ "$1" = "suggestive" ]; then
    wget -c https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/metaworld/model-4307.pt
    mkdir -p ckpts/metaworld
    mv model-4307.pt ckpts/metaworld/model-4307.pt
    echo "Downloaded VideoAgent-suggestive metaworld model"
elif [ "$1" = "ithor" ]; then
    wget -c https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/ithor/thor-402.pt
    mkdir -p ckpts/ithor
    mv model-402.pt ckpts/ithor/model-402.pt
    echo "Downloaded VideoAgent ithor model"
else 
    echo "Options: {metaworld, online, suggestive, ithor, bridge}"
fi
