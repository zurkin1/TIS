for file in Downloads/478*.tar
do
    #mv -i "${file}" "${file}".tar.gz
    tar xf "${file}" --skip-old-files &
    #gunzip "${file}" &
done
wait
