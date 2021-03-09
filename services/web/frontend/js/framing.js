
/* * * * * */
/*  DATA   */
/* * * * * */

let issue = '';
let trainingFile = '';
let newFrames = [];
let testingFile = '';
let emailAddress = '';


const help1 = 'Help panel 1';
const help2 = 'Help panel 2';
const help3 = 'Help panel 3';
const help4 = 'Help panel 4';


/* * * * * * */
/*  BROWSER  */
/* * * * * * */
$(document).ready(function() {

    // Step 1 page logic
    $("input[name='policyissue']").change(function(){
        // enable next if policy issue selected
        $("#nextBtn1").attr("disabled", false);
        // if choice is 'Other' show panel
        if ($(this).val() === 'other') {
            $('#other-issue').removeClass('hidden');
            $('#other-text').attr('disabled', false).focus();
        } else {
            $('#other-issue').addClass('hidden');
            $('#other-text').attr('disabled', true);
        }
    });

    $('#training-file-visible').on('click', function () {
        $('#training-file-invisible').click();
    });

    $("input[id='training-file-invisible']").change(function() {
        let file = $(this).val().split('\\').pop();
        $('#training-filepath')
            .html(`File chosen: ${file}`)
            .removeClass('hidden');
    });

    $('#example-training').click(function(e) {
        e.preventDefault();  //stop the browser from following
        window.location.href = 'examples/train.csv';

        $('#other-text').val('Gun Violence');
        $('#other-frames').val('2nd Amendment rights,Economic consequences,Gun control,Mental health,Politics,Public opinion,Race,School or public space safety,Society');
    });


    // Step 1 nav
    $('#backBtn1').on('click', function () {
        window.location.replace('client.html');
    });
    $('#nextBtn1').on('click', function () {
        $('#policy-issue').addClass('hidden');
        $('#data').removeClass('hidden');
        $('#help').html(help2);
        handlePolicyIssue();
    });


    // Step 2 page logic
    $('#testing-file-visible').on('click', function () {
        $('#testing-file-invisible').click();
    });

    $("input[id='testing-file-invisible']").change(function() {
        let file = $(this).val().split('\\').pop();
        $('#testing-filepath')
            .html(`File chosen: ${file}`)
            .removeClass('hidden');
    });

    $('#example-testing').click(function(e) {
        e.preventDefault();  //stop the browser from following
        window.location.href = 'examples/test.csv';
    });

    // Step 2 nav
    $('#backBtn2').on('click', function () {
        $('#data').addClass('hidden');
        $('#policy-issue').removeClass('hidden');
        $('#help').html(help1);
    });
    $('#nextBtn2').on('click', function () {
        $('#data').addClass('hidden');
        $('#email').removeClass('hidden');
        $('#help').html(help3);
    });


    // Step 3 nav
    $('#backBtn3').on('click', function () {
        $('#email').addClass('hidden');
        $('#data').removeClass('hidden');
        $('#help').html(help2);
    });
    $('#nextBtn3').on('click', function () {
        $('#email').addClass('hidden');
        $('#final').removeClass('hidden');
        $('#help').html(help4);
        setReviewPage();
    });


    // Step 4 nav
    $('#backBtn4').on('click', function () {
        $('#final').addClass('hidden');
        $('#email').removeClass('hidden');
        $('#help').html(help3);
    });

});





/* * * * * * * */
/*  HELPERS    */
/* * * * * * * */


function handlePolicyIssue() {
    if ($("input[name='policyissue']:checked").val() === 'other') {
        issue = $('#other-text').val();
        console.log(issue);
        newFrames = $('#other-frames').val().split(',');
        $('#review-issue').html(issue);
        trainingFile = $('#training-file-invisible').val();
        $('#review-training').html(trainingFile.split('\\').pop());
    } else {
        issue = $("input[name='policyissue']:checked").next('label').text();
        $('#review-training').html('N/A');
    }
}

function setReviewPage() {
    // issue global value is set in handlePolicyIssue()
    testingFile = $('#testing-file-invisible').val();
    emailAddress = $('#user-email').val();

    $('#review-issue').html(issue);
    $('#review-testing').html(testingFile.split('\\').pop());
    $('#review-email').html(emailAddress);
}


























