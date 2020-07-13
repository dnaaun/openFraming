/////////////
// HELPERS //
/////////////

function updateTopicModelContent(tmPrevGetData, context) {
    context.tmID = tmPrevGetData.topic_model_id;
    context.tmName = tmPrevGetData.topic_model_name;
    context.tmNumTopics = tmPrevGetData.num_topics;
    context.tmTopicNames = tmPrevGetData.topic_names;
    context.tmStatus = tmPrevGetData.status;
    if (tmPrevGetData.topic_previews !== undefined) {
        context.tmPreviews = tmPrevGetData.topic_previews;
        updatePreviewNames(tmPrevGetData.topic_names, context);
    } else {
        context.tmPreviews = [];
        // context.tmPreviews = [{name: 'name1', keywords:["k1", "k2"], examples:["this is an example", "this is another example"]},
        //     {name: 'name2', keywords:["k3", "k4"], examples:["a third example", "a fourth example"]}];
    }

}

function validNameInput (inputString, previews) {
    let inputArr = inputString.split(',');
    return inputArr.length === previews.length;
}

function updatePreviewNames(names, context) {
    for (let i = 0; i < context.tmPreviews.length; i++) {
        context.tmPreviews[i].name = names[i];
    }
}

function extractKeywordsFromPreviews(previews) {
    let keywords = [["Keywords"]];
    for (let p of previews) {
        for (let k of p.keywords) {
            keywords.push([k]);
        }
    }
    return keywords;
}



// doc ready
$(function() {
    ///////////////
    // CONSTANTS //
    ///////////////
    const BASE_URL = "http://www.openframing.org/api";

    //////////////////
    // HBS TEMPLATE //
    //////////////////
    let topicModelTemplate = $('#topic-model-spec-template').html();
    let topicModelTemplateScript = Handlebars.compile(topicModelTemplate);
    let topicModelContext = {tmID: "", tmName: "", tmNumTopics: "", tmStatus: "", tmPreviews:[]};
    let topicModelHtml = topicModelTemplateScript(topicModelContext);
    $('#topic-model-preview').append(topicModelHtml);

    // click event for submit button on topicModelPreviews.html
    $('#topic-model-preview-submit').on('click', function () {
        $('#no-spec-id').attr('hidden', true);
        $('#n-exist-id').attr('hidden', true);
        $('#no-keywords').attr('hidden', true);
        $('#no-proportions').attr('hidden', true);
        if ($('#topic-model-id').val() === "") {
            $('#no-spec-id').removeAttr('hidden');
        } else {
            $('#no-spec-id').attr('hidden', true);
            $('#n-exist-id').attr('hidden', true);
            $('#no-keywords').attr('hidden', true);
            $('#no-proportions').attr('hidden', true);
            const GET_ONE_TOPIC_MDL = BASE_URL + `/topic_models/${$('#topic-model-id').val()}/topics/preview`;

            $.ajax({
                url: GET_ONE_TOPIC_MDL,
                type: 'GET',
                dataType: 'json',
                success: function(data) {
                    console.log(data);
                    updateTopicModelContent(data, topicModelContext);
                    topicModelTemplate = $('#topic-model-preview').html();
                    topicModelHtml = topicModelTemplateScript(topicModelContext);
                    $('#topic-model-preview').empty().append(topicModelHtml);
                    $('#topic-model-preview').removeAttr('hidden');
                    $('#prev-specific-fields').removeAttr('hidden');
                },
                error: function(err) {
                    console.log(err);
                    $('#n-exist-id').removeAttr('hidden');
                }
            });
        }
    });

    $('#topic-names-submit').on('click', function () {
        $('#wrong-names').attr('hidden', true);
        if ($('#topic-names').val() === "") {
            $('#wrong-names').removeAttr('hidden');
        } else if (!validNameInput($('#topic-names').val(), topicModelContext.tmPreviews)) {
            $('#wrong-names').removeAttr('hidden');
        } else {
            $('#wrong-names').attr('hidden', true);
            const POST_TOPIC_NAMES = BASE_URL + `/topic_models/${topicModelContext.tmID}/topics/names`;

            let postData = {
                topic_names: $('#topic-names').val().split(',')
            };

            $.ajax({
                url: POST_TOPIC_NAMES,
                type: 'POST',
                dataType: 'json',
                data: postData,
                success: function(data) {
                    updateTopicModelContent(data, topicModelContext);
                    topicModelTemplate = $('#topic-model-spec-template').html();
                    topicModelTemplateScript = Handlebars.compile(topicModelTemplate);
                    topicModelHtml = topicModelTemplateScript(topicModelContext);
                    $('#topic-model-preview').empty().append(topicModelHtml);
                    $('#topic-model-preview').removeAttr('hidden');
                },
                error: handle_ajax_error
            });
        }
    });

    $('#keywords-submit').on('click', function () {
        // makeshift xls creation out of keywords
        let result_table = extractKeywordsFromPreviews(topicModelContext.tmPreviews);
        let lineArray = [];
        result_table.forEach(function(infoArray, index) {
            let line = infoArray.join(" \t");
            lineArray.push(index == 0 ? line : line);
        });
        let csvContent = lineArray.join("\r\n");
        let excel_file = document.createElement('a');
        excel_file.setAttribute('href', 'data:application/vnd.ms-excel;charset=utf-8,' + encodeURIComponent(csvContent));
        excel_file.setAttribute('download', `topic_model_${topicModelContext.tmName}_keywords.xls`);
        document.body.appendChild(excel_file);
        excel_file.click();
        document.body.removeChild(excel_file);

        // actual AJAX call to get the keywords
        // let GET_KEYWORDS = BASE_URL + `/topic_models/${topicModelContext.tmID}/keywords`;
        // $.ajax({
        //     url: GET_KEYWORDS,
        //     type: 'GET',
        //     data: {
        //         file_type: 'xlsx'
        //     },
        //     dataType: 'json',
        //     success: function(data) {
        //         console.log(data);
        //
        //     },
        //     error: function (err) {
        //         console.log(err);
        //         $('#no-keywords').removeAttr('hidden');
        //     }
        // });
    });

    $('#proportions-submit').on('click', function () {
        let GET_PROPORTIONS = BASE_URL + `/topic_models/${topicModelContext.tmID}/topics_by_doc`;

        $.ajax({
            url: GET_PROPORTIONS,
            type: 'GET',
            dataType: 'json',
            success: function(data) {
                console.log(data);
            },
            error: function (err) {
                console.log(err);
                $('#no-proportions').removeAttr('hidden');
            }
        });
    });
});
