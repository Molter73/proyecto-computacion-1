.PHONY: all
all: api apache ui

.PHONY: api
api:
	docker build --tag quay.io/mmoltras/classifiers:api \
		-f $(CURDIR)/api.Dockerfile \
		$(CURDIR)/../api

.PHONY: apache
apache:
	docker build --tag quay.io/mmoltras/classifiers:apache \
		-f $(CURDIR)/apache.Dockerfile \
		$(CURDIR)/../deploy

.PHONY: ui
ui:
	docker build --tag quay.io/mmoltras/classifiers:ui \
		-f $(CURDIR)/ui.Dockerfile \
		$(CURDIR)/../interfaz
