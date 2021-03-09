

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
            window.location.replace('framing.html');
        } else if (workflowChoice === 'lda') {
            window.location.replace('lda.html');
        } else {

        }
    })

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



















































































































































