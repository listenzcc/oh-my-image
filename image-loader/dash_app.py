# %%
import datetime

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

# %%
from my_image import MyImage

# %%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# %%
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '80%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])


def parse_contents(contents, filename, date):

    my_image = MyImage(src=contents, name=filename)

    return html.Div([
        # Section
        # Filename and date
        html.Hr(),
        html.Div(filename),
        html.Div(datetime.datetime.fromtimestamp(date)),

        # Section
        # Raw image
        html.Hr(),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, style={
                 'max-width': '600px', 'max-height': '600px'}),

        # Section
        # Raw content
        html.Hr(),
        html.Div('Raw content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


# %%
@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


# %%
if __name__ == '__main__':
    app.run_server(debug=True, port=8000)

# %%
