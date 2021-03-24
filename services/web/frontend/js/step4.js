
/* * * * * */
/*  DATA   */
/* * * * * */


function cleanCategories(categories) {
    let catsArr = categories.split(',');
    catsArr = catsArr.map((cat) => {return cat.trim()});
    return catsArr;
}


/* * * * * * */
/*  BROWSER  */
/* * * * * * */
$(document).ready(function() {

    $('#fc-training-visible').on('click', function () {
        $('#fc-training-invisible').click();
    });

    $("input[id='fc-training-invisible']").change(function() {
        let file = $(this).val().split('\\').pop();
        $('#fc-training-filepath')
            .html(`File chosen: ${file}`)
            .removeClass('hidden');
    });

    $('#submit4').on('click', function () {
        // handle missing info first
        if ($('#fc-name').val() === "") {
            $('#error4-text').html('Please provide a name for your classifier.');
            $('#error4').removeClass('hidden');
        } else if (document.getElementById("fc-training-invisible").files.length === 0) {
            $('#error4-text').html('Please provide a training file.');
            $('#error4').removeClass('hidden');
        } else if ($('#fc-labels').val() === "") {
            $('#error4-text').html('Please provide categories for your classifier.');
            $('#error4').removeClass('hidden');
        } else if ($('#fc-email').val() === "") {
            $('#error4-text').html('Please provide an email to send your results to.');
            $('#error4').removeClass('hidden');

        } else {

            $('#error4').addClass('hidden');

            // POST request for topic model
            const POST_CLASSIFIER = `${BASE_URL}/classifiers/`;
            const categories = cleanCategories($('#fc-labels').val());
            let postData = {
                classifier_name: $('#fc-name').val(),
                category_names: categories,
                notify_at_email: $('#fc-email').val()
            };
            $.ajax({
                url: POST_CLASSIFIER,
                type: 'POST',
                dataType: 'json',
                contentType: 'application/json',
                data: JSON.stringify(postData),
                success: function (data) {
                    console.log('success in topic model POST');
                    // POST request for training file
                    const POST_FC_TRAINING_FILE = `${BASE_URL}/topic_models/${data.classifier_id}/training/file`;
                    let fileFD = new FormData();
                    fileFD.append('file', document.getElementById("fc-training-invisible").files[0]);

                    $.ajax({
                        url: POST_FC_TRAINING_FILE,
                        data: fileFD,
                        type: 'POST',
                        processData: false,
                        contentType: false,
                        success: function(){
                            console.log('STEP 4 - success in training file POST');
                            $('#success4').removeClass('hidden');
                        },
                        error: function (xhr, status, err) {
                            console.log(xhr.responseText);
                            let error = getErrorMessage(JSON.parse(xhr.responseText).message);
                            $('#error4').html(`An error occurred while uploading your file: ${error}`).removeClass('hidden');
                        }
                    });
                },
                error: function (xhr, status, err) {
                    console.log(xhr.responseText);
                    let error = getErrorMessage(JSON.parse(xhr.responseText).message);
                    $('#error4').html(`An error occurred while creating the classifier: ${error}`).removeClass('hidden');
                }
            });
        }
    });

});


/* * * * * * * */
/*  HELPERS    */
/* * * * * * * */

