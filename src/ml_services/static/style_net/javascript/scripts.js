Dropzone.options.imageDropzone = {
    init: function(){
        this.on('success', function(err, response){
            console.log(response);
        });
    }
};