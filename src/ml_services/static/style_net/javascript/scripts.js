Dropzone.options.imageDropzone = {
    init: function(){
        this.on('success', function(err, response){
            window.location.href = "get?file="+response;
        });
    }
};