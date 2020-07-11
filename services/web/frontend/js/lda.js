$(function() {
    ///////////////
    // CONSTANTS //
    ///////////////
    const BASE_URL = "http://ec2-3-90-135-165.compute-1.amazonaws.com/api";

    //////////////////
    // HBS TEMPLATE //
    //////////////////
    let ldaTemplate = $('#topic-model-template').html();
    let ldaTemplateScript = Handlebars.compile(ldaTemplate);
    let ldaContext = {topicModelName : "", numTopics : ""};
    let ldaHtml = ldaTemplateScript(ldaContext);
    $('#all_topic_models').append(ldaHtml);

    // click event for submit button on LDA.html
    $('#topic-model-submit').on('click', function () {
        $('#no-name').attr('hidden', true);
        $('#no-number').attr('hidden', true);
        $('#no-file').attr('hidden', true);
        if ($('#topic-model-name').val() === "") {
            $('#no-name').removeAttr('hidden');
        } else if ($('#num-topics').val() === "") {
            $('#no-number').removeAttr('hidden');
        } else if (document.getElementById("ldatrainingfile").files.length === 0) {
            $('#no-file').removeAttr('hidden');
        } else {
            $('#no-name').attr('hidden');
            $('#no-number').attr('hidden');
            $('#no-file').attr('hidden');

            // POST request for topic model
            const POST_TOPIC_MODEL = BASE_URL + "/topic_models/";
            let postData = {
                topic_model_name: $('#topic-model-name').val(),
                num_topics: $('#num-topics').val()
            };
            $.ajax({
                url: POST_TOPIC_MODEL,
                type: 'POST',
                dataType: 'json',
                data: postData,
                success: function (data) {
                    console.log('success in first POST');

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
                        success: console.log('success on file POST'),
                        error: handle_ajax_error
                    });
                },
                error: handle_ajax_error
            });
        }
    });
});
