let BASE_URL = "http://" + window.location.host + "/api";
// uncomment below to test on AWS EC2 instance
// const BASE_URL = "http://ec2-3-90-135-165.compute-1.amazonaws.com/api";

/* * * * * * */
/*  BROWSER  */
/* * * * * * */
$(document).ready(function() {

    $("#step1").load("components/step1.html", function () {
        $.getScript("js/client.js");
        $.getScript("js/step1.js");
    });

    $("#step2").load("components/step2.html");

    $("#step3").load("components/step3.html");

    $("#step4").load("components/step4.html",function(){
        $.getScript("js/client.js");
        $.getScript("js/step4.js");
    });

    $("#step5").load("components/step5.html",function(){
        // $.getScript("js/step4.js");
    });


});

/* * * * * * */
/*  HELPERS  */
/* * * * * * */

function getErrorMessage(message) {
    if (typeof message === "object") {
        let strArr = [];
        for (let key of Object.keys(message)) {
            strArr.push(message[key]);
        }
        return strArr.join('; ')

    } else {
        return message;
    }
}


/* * * * * * * */
/*  HIDE/SHOW  */
/* * * * * * * */
