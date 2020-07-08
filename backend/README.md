# Notes
## Error messages
The API documentation below is missing what the API returns in case the user
provides incorrectly formatted input (or an incorrectly formatted file).

We're trying to be as RESTful an API as possible, so the *HTTP status codes*
*will be in the 4xx range if there was an error.* All the documentation below
pertains to when the *HTTP status code is 200*.

Regarding the body of the error messages, here is an example of an error
message when the user provides incorrectly formatted input to the `POST /classifiers`  endpoint.
(It will make more sense once you read through the documentation for the `POST /classifiers` endpoint)

The body of the request could have been:
```python
{
  "name": "Gun violence",
  "category_names": [
    "Economic consequences"
  ]
}
```

The HTTP status code is `400`. The body of the response is.
```python
{
    "message": "must be at least two categories."
}
```

## File formats
Currently, only CSV uploads are accepted. XLSX will be supported soon.

## URL Prefix
You will have to prefix every endpoint with `/api/`.  If the server is
running on `http://localhost:5000`, the first endpoint below can be accessed by
doing `http://localhost:5000/api/classifiers/`,

# API Documentation

## List all classifiers.
This returns a list of the details about every classifier that is on the system.
### Endpoint
`GET /classifiers/`

### Request body
Empty. 

### Return body 
#### When successful
```python
[
  { 
     # Not to be shown to the user. But will be needed to identify the classifier in further API requests.
     "classifier_id": int,  

     # Something like "gun violence", or "climate change"
     "classifier_name": str,

     # Whether this classifier was provided by openFraming, or the user trianed it
     "provided_by_openFraming": bool,

     # For "gun violence", might be things like "Economic consequences", "Gun rights", "Gun control", "Poltiics". "Public opinion"
     "category_names": [str, ...],

     # Indicates the status of the training for this classifier.
     # "not_begun" means no training data was not uploaded and training has not begun.
     # "training" means the classifier is training right now.
     # "completed' means the classifier has completed training.
     # "error_encountered" means exactly that. The user should attempt to 
     # create another classifier and start trainiing again.
     "status": One_of("not_begun", "training", "error_encountered", "completed"),

     # If the "status" is "completed" above, then "metrics" will indicate the performance
     # of the classifier on a development set.
     # Otherwise, it will be NULL.
     "metrics": One_of(
        NULL,
        {
	  "macro_f1_score": float,
	  "macro_precision": float,
	  "macro_recall": float,
	  "accuracy": float
        }
      )
   },
  ...
]
```

### Status
Done. 


## Creates a classifier.
This allows one to create a new classifier on the backend, ie, to provide the name of 
the new classifier and what the categories should be.
### Endpoint
`POST /classifiers/`

### Request body
```python
{
  "classifier_name": str,
  "category_names": [str, ...]
}
```

### Return body 
#### When successful
```python
{
  # Classifier id of the newly created classifier.
  "classifier_id": int,

  "category_names": [str, ...],

  # Note here this will always be False because the user created this classifier,
  # it's not provided by openFraming.
  "provided_by_openFraming": False,

  # The classifier was just created. No  training data was uploaded yet.
  "status": "not_begun"

  # The classifier is not trained yet, it has no metrics.
  "metrics": NULL,
}
```

### Status
Done.


## Get details about one classifier.
This is an endpoint to get details about ONE classifier, as opposed to ALL the classifiers.
The return body is identical to one elment of the return body from the `GET /classifiers/` endpoint.
### Endpoint
`GET /classifiers/<classifier_id:int>`

### Request body
Empty.


### Return body 
#### When successful
```python
{ 
  "classifier_id": int, 
  "classifier_name": str,
  "provided_by_openFraming": bool,
  "category_names": [str, ...],
  "status": One_of("not_begun", "training", "error_encountered", "completed")
  "metrics": One_of(
     NULL,
     {
       "macro_f1_score": float,
       "macro_precision": float,
       "macro_recall": float,
       "accuracy": float
     }
   )
}
```

### Status
Done.


## Upload training data for a classifier and start training.
Once a classifier has been created(ie, the name of the classifier, and the categories names are created 
in the backend using the `POST /classifiers/`, the user will have to provide training data to train the classifier.

### Endpoint
`POST /classifiers/<classifier_id:int>/training/file`

### Request body
`FormData` 
 `file`: A CSV/Excel file.

#### File format
The file has to to be  CSV or an Excel that looks like the following. The
headers are mandatory as well.

| Example                                                                                | Category               |
| -------------------------------------------------------------------------------------- | ---------------------- |
| Anti-gun student walkout included stomping on American flag and jumping on cop car     | Public opinion         |
| Parkland students brand firearm ban during Pence speech as NRA hypocrisy               | Public opinion         |
| Parkland survivors keep memory of shooting alive                                       | Public opinion         |
| What is gun control? Everything you need to know                                       | Gun control/regulation |
| Remington, Centuries-Old Gun Maker, Files for Bankruptcy as Sales Slow                 | Economic consequences  |
| Watch Live: Trump addresses NRA’s annual convention in Dallas                          | Politics               |
| Bernie Sanders lashes out at Congress over gun control after Santa Fe, Texas, shooting | Politics               |
| Trump's 'Angel Families' weaponize their grief to demonize immigrants                  | Politics               |

### Return body 
#### When successful
```python
{ 
  "classifier_id": int, 
  "classifier_name": str,
  "provided_by_openFraming": False,
  "category_names": [str, ...],

  # The classifier just began training.
  "status": "training",

  # metrics will be NULL because the classifier just started to train
  "metrics": NULL
}
```

### Status
Done.

## Lists all test sets for a classifier.
Once a classifier has been created and trained, one can do prediction on test sets with no labels using this endpoint.
### Endpoint
`GET /classifiers/<classifier_id:int>/test_sets/`

### Request body
Empty.

### Return body 
#### When successful
```python
[
  {
    # Th classifier id of the classifier that this test set is intended for.
    "classifier_id": int,

    # The id of this test set. Like with classifier_id in the /classifiers/
    # endpoint, it is used  used to identifyNote here the status will always be not_begun because t
    "test_set_id": int,
    "test_set_name": str,

    # The status here indicates whether inference on this test set has begun, is
    # in progress, or has completed.
    "status": One_of("not_begun", "predicting", "error_encountered", "completed")
  } ...
]
```

### Status
Done.

## Create a test set.
This will be used to "create a new test set", ie, to provide a name for a test
set. Note that we don't upload test data using this endpoint.
### Endpoint
`POST /classifiers/<classifier_id:int>/test_sets/`

### Request body
```python
{ 
  # For a gun violence classifier, might be something like
  # "Gun violence related headlinesfrom 2020", or "Gun violence related headlines from California"
  "test_set_name": str, 
}
```

### Return body 
#### When successful
```python
{ 
  "classifier_id": int,
  "test_set_id": int,
  "test_set_name": str,

  # No test data is uploaded yet.
  "status": "not_begun"
}
``` 

### Status
Done.

## Get details about one test set.
`GET /classifiers/<classifier_id:int>/test_sets/` provides details about
every test set for a specific classifier, this endpoint provides details about
one test set.
### Endpoint
`GET /classifiers/<classifier_id:int>/test_sets/<test_set_id:int>`

### Request body
Empty.


### Return body 
 #### When successful
```python
{
  "classifier_id": int,
  "test_set_id": int
  "test_set_name": str,
  "inference_status": One_of("not_begun", "predicting", "completed")
}
```

### Status
Done.


## Upload test set and start inference.
Once a test set is created, ie, the name was set by the user, the user can then
upload the test set data.

### Endpoint
`POST /classifiers/<classifier_id:int>/test_sets/<test_set_id:int>/file`

### Request body
`FormData`
1. `file`

#### File format
A CSV/Excel file with the following format. The header row is necessary.

| Example                                                                                       |
| ------------------------------------------------------------------------------------------------ |
| What is gun control? Everything you need to know                                                 |
| 17-year-old charged with murder in shooting of 11-year-old boy in East Chicago                   |
| Sepeda guilty of murdering Elgin man, kidnapping his girlfriend and baby at gunpoint, jury rules |
| 22 West Aurora students participate in third national walkout since Parkland shooting            |
| Central Michigan student used father's gun to kill his parents, police say                       |
| In Michigan, gun debate and governor's race collide                                              |
| 6 Things You Can Do Right Now to Fight for Gun-Control                                           |
| James Comey recalls a 'life-changing' run-in with a gunman who broke into his childhood home     |
| Teen Wearing Neo-Nazi Group Clothing Arrested With Cache of Illegal Weapons in Illinois          |

### Return body 
#### When successful
```python
{ 
  "classifier_id": int,
  "test_set_id": int,
  "test_set_name": str,

  # We just began predicting on this test set
  "status": "predicting"
}
```

### Status
Done.


## Download predictions on test set.
Once the user has uploaded a test set, if the prediction was completed, a user
can download the predictions on the test set data using this endpoint.
### Endpoint
`GET /classifiers/<classifier_id:int>/test_sets/<test_set_id:int>/predictions`

### Request parameters
1. `file_type`: Optional. Can be:`csv`, `xlsx`.

### Request body
Empty.

### Return body 
#### When successful
A CSV/Excel file with the predictions.


### Status
In progress.



# List all topic models.
### Endpoint
`GET /topic_models/`

### Request body
Empty.

### Return body 
#### When successful
```python
[
  {
    "topic_model_id": int,
    "topic_model_name": str,

    # The number of topics to provide to the topic model
    "num_topics": int,

    # The user will be able to provide names for the topics discovered by the topic model.
    # If the training finished and the user provided the names, this will not
    # be NULL.
    "topic_names": One_of(NULL, [str, ...])

    # "training" indicates the topic model is training currently. "topics_to_be_named"
    # indicates the topic model finished training, but the topics have not been named yet by the
    # user. "completed" indicates the topic model finished training, and the user assigned names to the topics.
    "status": One_of("not_begun", "training", "error_encountered", "topics_to_be_named", "completed")
  },
  ...
]
```

### Status
 Done



## Creates a topic model.
### Endpoint
`POST /topic_models/`

### Request body
```python
{
  # Something like "Coronavirus news coverage 2020" (to take an example from
  # what AIEM has been doing recently).
  "topic_model_name": str,

  # Number of topics to discover using the topic model.
  "num_topics": int
}
```

### Return body 
#### When successful
```python
{
  "topic_model_id": int, 
  "topic_model_name": str,
  "num_topics": int,
  "topic_names": NULL, 
  "status": "not_begun"
 }
```

### Status
Done.


## Get details about one topic model.
### Endpoint
`GET /topic_models/<topic_model_id:int>`

### Request body 
Empty.

### Return body 
#### When successful
```python
{
  "topic_model_id": int,
  "topic_model_name": str,
  "num_topics": int,
  "topic_names": One_of(NULL, [str, ...]),
  "status": One_of("not_begun", "training", "error_encountered", "topics_to_be_named", "completed")
}
```


### Status

## Upload training file and start training topic model.
Once a topic model is created, the user can upload a file to start training the
topic model.
### Endpoint
`POST /topic_models/<topic_model_id:int>/training/file`

### Request body
`FormData`

1. `file`
#### File format
A CSV/XLSX file with one column, with a header named "Example". The 
file format is identical to the file format required by the 
`/classifiers/<classifier_id:int>/test_sets/<test_set_id:int>/file` endpoint.

### Return body 
#### When successful
```python
{
  "topic_model_id": int,
  "topic_model_name": str,
  "num_topics": int,
  "topic_names": NULL,
  "status":  "training"
}
```

### Status
Done

## Get a preview of the topics discovered by the topic model.
Before assigning human readable names to the topics discovered by the topic
model, the user has to get a preview of the results of the topic model.

Note that the user can still invoke this endpoint even after assigning names.

### Endpoint
`GET /topic_models/<topic_model_id:int>/topics/preview`

### Request body
Empty.

### Return body 
#### When successful
```python
{
  "topic_model_id": int 
  "topic_model_name": str,
  "num_topics": int,

  "topic_names": One_of(NULL, [str, ...]),

  # If we're providing a preview, the topic model has been trained already,
  # which is why the status can only be one of the two below.
  "status":  One_of("topics_to_be_named", "completed"),

  # These are the topic previews for each topic discovered by the topic model.
  # The length of this list will be equal to `num_topics` above.
  # THe first item of the list corresponds to the first topic discovered by LDA,
  # and so on.
  "topic_previews": [
    {
      # The important keywords for each topic
      "keywords": [ str, ... ],

      # Examples (ie, documents) for which the topic has highest "responsibilty"
      # for
      # In LDA topic modeling, a document is produced by multiple topics.
      "examples": [str, ... ]
    },
    ...  
  ]
}
```

### Status
Done.


## Name the topics of a trained topic model.
Once the user gets a preview of the results of the topic model, they can provide
human readable names to the topic.

### Endpoint
`POST /topic_models/<topic_model_id:int>/topics/names`

### Request body
```python
{ 
    # Length must be equal to "num_topics" of topic model.
    # The first element of this list corresponds to the name given to the first
    # topic discovered by LDA, and so on.
    "topic_names": [str, ...] 
}
```

### Return body 
 #### When successful
```python
{
     "topic_model_id": int, 
     "topic_model_name": str,
     "num_topics": int,
     "topic_names": [str, ...],
     "status": "completed"
}
```

### Status
Done

## Download keywords per each topic.
### Endpoint
`GET /topic_models/<topic_model_id:int>/keywords`

### Request parameters
1. `file_type`: Optional. Can be:`csv`, `xlsx`.

### Request body
Empty.

### Return body 
#### When successful
A `csv` or `xlsx` file.


### Status
In progress.

## Download topic proportions for each example.
### Endpoint
`GET /topic_models/<topic_model_id:int>/keywords`

### Request parameters
1. `file_type`: Optional. Can be:`csv`, `xlsx`.

### Request body
Empty.

### Return body 
#### When successful
A `csv` or `xlsx` file.

### Status
In progress.
