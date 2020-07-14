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

/*
* Get url parameters.
* @param {string} name - Url parameter name
* @return {string} The value of the parameter, or 0.
*/
function getUrlParam(name) {
   var results = new RegExp("[?&]" + name + "=([^&#]*)").exec(
     window.location.href);
   if (results == null) {
     return null;
   }
 return decodeURI(results[1]);
}

// doc ready
$(function() {
    ///////////////
    // CONSTANTS //
    ///////////////
    const BASE_URL = "http://" + window.location.host + "/api";

    //////////////////
    // HBS TEMPLATE //
    //////////////////
    let topicModelTemplate = $('#topic-model-spec-template').html();
    let topicModelTemplateScript = Handlebars.compile(topicModelTemplate);
    let topicModelContext = {tmID: "", tmName: "", tmNumTopics: "", tmStatus: "", tmPreviews:[]};
    let topicModelHtml = topicModelTemplateScript(topicModelContext);
    $('#topic-model-preview').append(topicModelHtml);

   
      // Grab topic model id from url parameters
      var topic_model_id = parseInt(getUrlParam('topic_model_id'));
      if (isNaN(topic_model_id)) {
            $('#invalid-topic-model-id-msg').removeAttr('hidden');
        } else { 
          $('#invalid-topic-model-id-msg').attr('hidden', true);
          const GET_ONE_TOPIC_MDL = BASE_URL + `/topic_models/${topic_model_id}/topics/preview`;

          $.ajax({
              url: GET_ONE_TOPIC_MDL,
              type: 'GET',
              dataType: 'json',
              success: function(data, status) {
                  console.log(data);
                  updateTopicModelContent(data, topicModelContext);
                  topicModelTemplate = $('#topic-model-preview').html();
                  topicModelHtml = topicModelTemplateScript(topicModelContext);
                  $('#topic-model-preview').empty().append(topicModelHtml);
                  $('#topic-model-preview').removeAttr('hidden');
                  $('#prev-specific-fields').removeAttr('hidden');

                  $('#invalid-topic-model-id-msg').attr('hidden', true);
                  $('#server-error-msg').attr('hidden', true);
              },
              error: function(jqxhr) {
                  console.log("Error:", jqxhr);
                  if (jqxhr.status.toString()[0] == "5") { // 5xx errors
                      $('#invalid-topic-model-id-msg').attr('hidden', true);
                      $('#server-error-msg').removeAttr('hidden');
                  } else {
                      $('#invalid-topic-model-id-msg').removeAttr('hidden');
                      $('#server-error-msg').attr('hidden', true);
                  }

              }
          });
      }

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
                contentType: 'application/json',
                data: JSON.stringify(postData),
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
        let GET_KEYWORDS = BASE_URL + `/topic_models/${topicModelContext.tmID}/keywords?file_type=xlsx`;
        let a = document.createElement("a");
        a.href = GET_KEYWORDS;
        a.click();
    });

    $('#proportions-submit').on('click', function () {
        let GET_PROPORTIONS = BASE_URL + `/topic_models/${topicModelContext.tmID}/topics_by_doc?file_type=xlsx`;
        let a = document.createElement("a");
        a.href = GET_PROPORTIONS;
        a.click();
    });
});
