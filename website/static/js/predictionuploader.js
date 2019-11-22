$(document).ready(function() {
    $("#upload_form button").click(function (event) {

        event.preventDefault();
        console.log($(this).attr("value"))
        var formData = new FormData($('form')[$(this).attr("value")]);
        console.log(formData)
        $.ajax({
            xhr : function() {
                var xhr = new window.XMLHttpRequest();

                xhr.upload.addEventListener('progress', function(e) {

                    if (e.lengthComputable) {

                        console.log('Bytes Loaded: ' + e.loaded);
                        console.log('Total Size: ' + e.total);
                        console.log('Percentage Uploaded: ' + (e.loaded / e.total))

                        var percent = Math.round((e.loaded / e.total) * 100);

                        $('#progressBar').attr('aria-valuenow', percent).css('width', percent + '%').text(percent + '%');

                    }

                });

                return xhr;
            },
            type : 'POST',
            url : '/custommodels/',
            data : formData,
            processData : false,
            contentType : false,
            success: function(message, textStatus, response) {
       var header = response.getResponseHeader('Content-Disposition');
       if (header && header.indexOf('attachment') !== -1) {
            var filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
            var matches = filenameRegex.exec(header);
            if (matches != null && matches[1]) filename = matches[1].replace(/['"]/g, '');
        }
       var fileName = decodeURI(filename);
       var blob = new Blob([message]);
       var link = document.createElement('a');
       link.href = window.URL.createObjectURL(blob);
       link.download = fileName;
       link.click();
    }
        });

       $("#upload_form")[$(this).attr("value")].reset();
    });

});