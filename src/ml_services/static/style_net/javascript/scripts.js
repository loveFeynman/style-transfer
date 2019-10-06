//Dropzone.autoDiscover = false;

var options = {
    init: function(){
        this.on('success', function(err, response){
            console.log('response');
            window.location.href = "get?file="+response;
        });
    }
};

//var z1 = document.getElementById('rain-dropzone');
//var z2 = document.getElementById('wave-dropzone');
//var z3 = document.getElementById('starry-dropzone');
//
//var dz1 = new Dropzone('rain-dropzone', options);
//var dz2 = new Dropzone('wave-dropzone', options);
//var dz3 = new Dropzone('starry-dropzone', options);

Dropzone.options.rainDropzone = options;
Dropzone.options.waveDropzone = options;
Dropzone.options.starryDropzone = options;

// Dropzone.options.imageDropzone = options;