
/* * * * * */
/*  DATA   */
/* * * * * */



/* * * * * * */
/*  BROWSER  */
/* * * * * * */
$(document).ready(function() {

    $('#tm-training-visible').on('click', function () {
        $('#tm-training-invisible').click();
    });

    $("input[id='tm-training-invisible']").change(function() {
        let file = $(this).val().split('\\').pop();
        $('#tm-training-filepath')
            .html(`File chosen: ${file}`)
            .removeClass('hidden');
    });

    $('#backBtn').on('click', function () {
        window.location.replace('client.html');
    });

});




/* * * * * * * */
/*  HELPERS    */
/* * * * * * * */

























