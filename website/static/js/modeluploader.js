$(document).ready(function() {

    $('form').on('submit', function(event) {
        $('#records_table').empty()
        event.preventDefault();

        var formData = new FormData($('form')[0]);

        $.ajax({
            xhr : function() {
                var xhr = new window.XMLHttpRequest();

                xhr.upload.addEventListener('progress', function(e) {

                    if (e.lengthComputable) {

                        console.log('Bytes Loaded: ' + e.loaded);
                        console.log('Total Size: ' + e.total);
                        console.log('Percentage Uploaded: ' + (e.loaded / e.total));

                        var percent = Math.round((e.loaded / e.total) * 100);

                        $('#progressBar').attr('aria-valuenow', percent).css('width', percent + '%').text(percent + '%');

                    }

                });

                return xhr;
            },
            type : 'POST',
            url : '/train/',
            data : formData,
            processData : false,
            contentType : false,
            success : function(data) {
                console.log(data);
                response = JSON.parse(data);
                var trHTML = '';
                var flag=1;
                $.each(response, function (i, item) {
                    obj=item;
                    if(flag==1)
                        trHTML += ' <thead><tr">';
                    else
                        trHTML += '<tbody"><tr>';
                    for (var key in obj) {
                      if (obj.hasOwnProperty(key)) {
                        var val = obj[key];
                        trHTML += '<td>' + val + '</td>';

                      }
                    }
                    if(flag==1)
                        trHTML += '</tr></thead>';
                    else
 
                        trHTML += '</tr></tbody>';
                    flag=0;
                });
                trHTML += '<tr><td colspan="100"><b>These are some sample data from your dataset. Your Model is being Trained. Please Check your Email or Click the Notification Icon for More Details.</td></tr></b>';
                $('#records_table').append(trHTML);
        }
      });

    });

});
 