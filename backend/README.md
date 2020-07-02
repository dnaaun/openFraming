NOTE: You will have to prefix every endpoint with "/api/".  If the server is running on http://localhost:5000, the first endpoint below can be accessed by doing http://localhost:5000/api/classifiers


| <!-- -->                             | <!-- -->                                                                                                                                                                                                               |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Endpoint                             | /classifiers/                                                                                                                                                                                                          |
| Method                               | GET                                                                                                                                                                                                                    |
| Request body                         |                                                                                                                                                                                                                        |
| Return body <br>(in case of success) | [<br>{ <br> "classifier_id": int, <br>"policy_issue_name": str,<br>"provided_by_openFraming": bool,<br>"frame_names": [str, ...],<br>"training_status": One_of("not_begun", "training", "completed")<br>},<br>...<br>]
| Remarks                              | Lists all classifiers.                                                                                                                                                                                                 |
| Status                               | Done                                                                                                                                                                                                                   |



| <!-- -->                             | <!-- -->                                                                                                                                                         |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Endpoint                             | /classifiers/                                                                                                                                                    |
| Method                               | POST                                                                                                                                                             |
| Request body                         | {<br>"policy_issue_name": str,<br>"frame_names": [str, ...]<br>}                                                                                                 |
| Return body <br>(in case of success) | {<br>"classifier_id": int,<br>"policy_issue_name": str,<br>"provided_by_openFraming": bool,<br>"frame_names": [str, ...],<br>"training_status": "not_begun"<br>} |
| Remarks                              | Creates a frame classifier                                                                                                                                       |
| Status                               | Done                                                                                                                                                             |


| <!-- -->                              | <!-- -->                                                                                                                                               |
|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Endpoint                              | /classifiers/<int:classifier_id>                                                                                                                       |
| Method                                | GET                                                                                                                                                    |
| Request body                          |                                                                                                                                                        |
| Return body <br> (in case of success) | {<br>"classifier_id": int,<br>"name": str,<br>"trained_by_openFraming": false,<br>"training_status": One_of("not_begun", "training", "completed")<br>} |
| Remarks                               | Gets details about a classifier                                                                                                                        |
| Status                                | In progress                                                                                                                                            |

| <!-- -->                              | <!-- -->                                                                                                 |
|---------------------------------------|----------------------------------------------------------------------------------------------------------|
| Endpoint                              | /classifiers/<classifier_id:int>/training/file                                                           |
| Method                                | POST                                                                                                     |
| Request body                          | FormData <br> "file": A CSV/Excel file.                                                                  |
| Return body <br> (in case of success) | -                                                                                                        |
| Remarks                               | This will start training the model. We will split this file into a training and test set in the backend. |
| Status                                | Done |


| <!-- -->                              | <!-- -->                                                                                                 |
|---------------------------------------|----------------------------------------------------------------------------------------------------------|
| Endpoint                              | /classifiers/<int:classifier_id>/training/metrics                                                        |
| Method                                | GET                                                                                                      |
| Request body                          | -                                                                                                        |
| Return body <br> (in case of success) | {<br>"macro_f1": float,<br>"macro_precision": float,<br>"macro_recall": float,<br>"accuracy": float<br>} |
| Remarks                               | Gets metrics of the classifier                                                                           |
| Status                                |                                                                                                          |

| <!-- -->                              | <!-- -->                                                                                                                         |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Endpoint                              | /classifiers/<classifier_id:int>/prediction_set/                                                                                 |
| Method                                | POST                                                                                                                             |
| Request body                          | { <br> "set_name": str, # Some name the user wants to provide for this set<br>}                                                  |
| Return body <br> (in case of success) | { <br>"classifier_id": int,<br>"prediction_set_id": int,<br>"prediction_set_name": str,<br>"inference_status": "predicting"<br>} |
| Remarks                               | Starts doing inference on a bunch of examples                                                                                    |
| Status                                |                                                                                                                                  |


| <!-- -->                              | <!-- -->                                                                                                                                                            |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Endpoint                              | /classifiers/<classifier_id:int>/prediction_sets/                                                                                                                   |
| Method                                | GET                                                                                                                                                                 |
| Request body                          | -                                                                                                                                                                   |
| Return body <br> (in case of success) | [<br>{<br>"classifier_id": int,<br>"prediction_set_id": int,<br>"prediction_set_name": str,<br>"inference_status": One_of("predicting", ""completed")<br>} ...<br>] |
| Remarks                               | Lists all <b>batched prediction sets</b> for classifier                                                                                                             |
| Status                                |                                                                                                                                                                     |



| <!-- -->                              | <!-- -->                                                                                                                                             |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| Endpoint                              | /classifiers/<classifier_id:int>/prediction_sets/<int:prediction_set_id>                                                                             |
| Method                                | GET                                                                                                                                                  |
| Request body                          | -                                                                                                                                                    |
| Return body <br> (in case of success) | {<br>"classifier_id": int,<br>"prediction_set_id": int<br>"prediction_set_name": str,<br>"inference_status": One_of("predicting", ""completed")<br>} |
| Remarks                               | Gets details about a prediction set                                                                                                                  |
| Status                                |                                                                                                                                                      |

| <!-- -->                              | <!-- -->                                                                      |
|---------------------------------------|-------------------------------------------------------------------------------|
| Endpoint                              | /classifiers/<classifier_id:int>/prediction_sets/<int:prediction_set_id>/file |
| Method                                | POST                                                                          |
| Request body                          | FormData<br>1. "file"                                                         |
| Return body <br> (in case of success) | -                                                                             |
| Remarks                               | Starts doing inference on a prediction set                                    |
| Status                                |                                                                               |

| <!-- -->                              | <!-- -->                                                                            |
|---------------------------------------|-------------------------------------------------------------------------------------|
| Endpoint                              | /classifiers/<classifier_id:int>/prediction_sets/<int:prediction_set_id>/prediction |
| Method                                | GET                                                                                 |
| Request body                          | -                                                                                   |
| Return body <br> (in case of success) | A CSV file with the predictions                                                     |
| Remarks                               | Gets the predictions for the prediction set                                         |
| Status                                |                                                                                     |


| <!-- -->                               | <!-- -->                                                                                                                                                                                            |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Endpoint                               | /topic_models/                                                                                                                                                                                      |
| Method                                 | GET                                                                                                                                                                                                 |
| Request body                           | -                                                                                                                                                                                                   |
| Return body <br> ( in case of success) | [<br>{<br>"topic_model_id": int,<br>"topic_model_name": str,<br>"num_topics": int,<br>"topic_names": One_of(NULL, [str, ...])<br>"status": One_of("not_begun", "training", "topics_to_be_named", "completed")<br>}, ...<br>] |
| Remarks                                | Lists all topic models currently created                                                                                                                                                            |
| Status                                 |                                                                                                                                                                                                     |


| <!-- -->                              | <!-- -->                                                                                                                                                            |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Endpoint                              | /topic_models/                                                                                                                                                      |
| Method                                | POST                                                                                                                                                                |
| Request body                          | {<br>"topic_model_name": str,<br>"num_topics": int # The desired number of topics<br>}                                                                              |
| Return body <br> (in case of success) | {<br>"topic_model_id": int # The id of the topic detector just created<br>"topic_model_name": str,<br>"num_topics": int,<br>"topic_names": NULL, <br>"status": "training"<br> } |
| Remarks                               | Creates a topic classifier                                                                                                                                          |
| Status                                |                                                                                                                                                                     |



| <!-- -->                              | <!-- -->                                                                                                                                                                             |
|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Endpoint                              | /topic_models/<topic_model_id:int>                                                                                                                                                                       |
| Method                                | GET                                                                                                                                                                                  |
| Request body                          | |
| Return body <br> (in case of success) | {<br>"topic_model_id": int,<br>"topic_model_name": str,<br>"num_topics": int,<br>"topic_names": One_of(NULL, [str, ...])<br>"status": One_of("not_begun", "training", "topics_to_be_named", "completed")<br>} |
| Remarks                               | |
| Status                                |                                                                                                                                                                                      |


| <!-- -->                              | <!-- -->                                         |
|---------------------------------------|--------------------------------------------------|
| Endpoint                              | /topic_models/<topic_model_id:int>/training/file |
| Method                                | POST                                             |
| Request body                          | FormData<br>1. "file"                            |
| Return body <br> (in case of success) |                                                  |
| Remarks                               | Begins <b>training</b> the topic detector        |


| <!-- -->                              | <!-- -->                                                                                                                                                                                                                                                                        |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Endpoint                              | /topic_models/<topic_model_id:int>/topics/preview                                                                                                                                                                                                                               |
| Method                                | GET                                                                                                                                                                                                                                                                             |
| Request body                          | -                                                                                                                                                                                                                                                                               |
| Return body <br> (in case of success) | {<br>"topic_model_id": int <br>"topic_model_name": str,<br>"num_topics": int,<br>"topic_names": One_of(NULL, [str, ...])<br>"topic_previews": [<br>{<br>"important_words": [ str, ... ]<br>"examples": [str, ... ]<br>} ...<br># The length of this list is equal to "num_topics"<br>]<br>} |
| Remarks                               | Gets a preview of each topic                                                                                                                                                                                                                                                    |
| Status                                |                                                                                                                                                                                                                                                                                 |



| <!-- -->                              | <!-- -->                                                                                                                                                                                                                                                           |
|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Endpoint                              | /topic_models/<topic_model_id:int>/topics/names                                                                                                                                                                                                                    |
| Method                                | POST                                                                                                                                                                                                                                                               |
| Request body                          | <pre>{ <br>    "topic_names": [str, ...] # Length must be equal to "num_topics" <br>} </pre>|
| Return body <br> (in case of success) | <pre>{<br>     "topic_model_id": int <br>     "topic_model_name": str,<br>     "num_topics": int,<br>     "topic_names": [str, ...],<br>      "status": "completed"<br>} |
| Remarks                               | Names the topics                                                                                                                                                                                                                                                   |
| Status                                |                                                                                                                                                                                                                                                                    |



| <!-- -->                              | <!-- -->                                                                                                                                                 |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Endpoint                              | /topic_models/<topic_model_id:int>/samples?<num_samples:int>                                                                                             |
| Method                                | GET                                                                                                                                                      |
| Return body <br> (in case of success) | A CSV file that contains samples for each topic.                                                                                                         |
| Remarks                               | One can upload this CSV file to the /classifiers/... endpoint to begin training.<br>Question: We're going to assign the primary topic for each document. |
| Status                                |                                                                                                                                                          |
