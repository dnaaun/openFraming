$(function() {
    ///////////////
    // CONSTANTS //
    ///////////////
    const BASE_URL = "http://" + window.location.host + "/api";
    // const BASE_URL = "http://ec2-3-90-135-165.compute-1.amazonaws.com/api";

    //////////////////
    // HBS TEMPLATE //
    //////////////////
    let ldaTemplate = $('#topic-model-template').html();
    let ldaTemplateScript = Handlebars.compile(ldaTemplate);
    let ldaContext = {topicModelName : "", numTopics : "", notifyEmail: "", trainingDataLanguage: ""};
    let ldaHtml = ldaTemplateScript(ldaContext);
    $('#all_topic_models').append(ldaHtml);

    // click event for submit button on LDA.html
    $('#topic-model-submit').on('click', function () {
        $('#no-name').attr('hidden', true);
        $('#no-number').attr('hidden', true);
        $('#no-file').attr('hidden', true);
        $('#no-email').attr('hidden', true);
        $('#err-creating-tm').attr('hidden', true);
        $('#err-uploading').attr('hidden', true);
        $('#tm-success-message').attr('hidden', true);
        if ($('#topic-model-name').val() === "") {
            $('#no-name').removeAttr('hidden');
        } else if ($('#tm-num-topics').val() === "") {
            $('#no-number').removeAttr('hidden');
        } else if (document.getElementById("ldatrainingfile").files.length === 0) {
            $('#no-file').removeAttr('hidden');
        } else if ($('#tm-notify-email').val() === "") {
            $('#no-email').removeAttr('hidden');
        } else {
            $('#no-name').attr('hidden', true);
            $('#no-number').attr('hidden', true);
            $('#no-file').attr('hidden', true);
            $('#no-email').attr('hidden', true);

            // POST request for topic model
            const POST_TOPIC_MODEL = BASE_URL + "/topic_models/";
            let postData = {
                topic_model_name: $('#topic-model-name').val(),
                num_topics: $('#tm-num-topics').val(),
                notify_at_email: $('#tm-notify-email').val()
            };
            $.ajax({
                url: POST_TOPIC_MODEL,
                type: 'POST',
                dataType: 'json',
                contentType: 'application/json',
                data: JSON.stringify(postData),
                success: function (data) {
                    console.log('success in topic model POST');

                    // POST request for training file
                    const POST_TM_TRAINING_FILE = BASE_URL + `/topic_models/${data.topic_model_id}/training/file`;
                    let fileFD = new FormData();
                    fileFD.append('file', document.getElementById("ldatrainingfile").files[0]);

                    $.ajax({
                        url: POST_TM_TRAINING_FILE,
                        data: fileFD,
                        type: 'POST',
                        processData: false,
                        contentType: false,
                        success: function(){
                            console.log('success in training file POST');
                            $('#tm-success-message').removeAttr('hidden');
                        },
                        error: function (xhr, status, err) {
                            console.log(xhr.responseText);
                            let error = getErrorMessage(JSON.parse(xhr.responseText).message);
                            // if (error.num_topics) {
                            //     $('#err-creating-tm').html(`An error occurred while uploading your file: ${error.num_topics}`)
                            //         .removeAttr('hidden');
                            // } else{
                                $('#err-creating-tm').html(`An error occurred while uploading your file: ${error}`)
                                    .removeAttr('hidden');
                            // }
                        }
                    });
                },
                error: function (xhr, status, err) {
                    console.log(xhr.responseText);
                    let error = getErrorMessage(JSON.parse(xhr.responseText).message);
                    // if (error.num_topics) {
                    //     $('#err-creating-tm').html(`An error occurred while creating the topic model: ${error.num_topics}`)
                    //         .removeAttr('hidden');
                    // } else{
                        $('#err-creating-tm').html(`An error occurred while creating the topic model: ${error}`)
                            .removeAttr('hidden');
                    // }

                }
            });
        }
    });
});
