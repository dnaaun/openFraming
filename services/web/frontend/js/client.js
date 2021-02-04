

let workflowChoice = '';

/* * * * * * */
/*  BROWSER  */
/* * * * * * */
$(document).ready(function() {

    // show workflow options if click "Playground"
    $('#playground').on('click', function () {
        showWorkflowOptions();
    });

    // to go back to landing page from workflow options
    $('#backBtn').on('click', function () {
        backToLanding();
    });

    // enable Next if workflow is selected
    $('input[type=radio][name=workflow]').change(function() {
        $("#nextBtn").attr("disabled", false);
        workflowChoice = $(this).val();
    });

    $('#nextBtn').on('click', function () {
        if (workflowChoice === 'framing') {
            window.location.replace('step4.html');
        } else if (workflowChoice === 'lda') {
            window.location.replace('step1.html');
        } else {

        }
    });

    $("#step1").load("step1.html", function () {
        $.getScript("js/step1.js");
    });

    $("#step4").load("step4.html",function(){
        $.getScript("js/step4.js");
    });

    $("#step5").load("step5.html",function(){
        $.getScript("js/step4.js");
    });


});



/* * * * * * * */
/*  HIDE/SHOW  */
/* * * * * * * */


function showWorkflowOptions() {
    $('#landing').addClass('hidden');
    $('#workflow-choice').removeClass('hidden');
}

function backToLanding() {
    $('#workflow-choice').addClass('hidden');
    $('#landing').removeClass('hidden');
}

